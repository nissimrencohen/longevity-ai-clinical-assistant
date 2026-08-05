"""Inbound PHI de-identification: names never reach the external model.

THE PROBLEM THIS SOLVES, AND WHY IT LOOKED UNSOLVABLE. Every earlier phase could
control what the TOOLS return, but not what the doctor types. "What is Maya
Cohen's eGFR?" goes to OpenRouter verbatim, and controlling that needs something
that owns the inbound turn — which LibreChat's built-in agent is not.

The guard proxy turns out to be exactly that thing. It already sits between
LibreChat and OpenRouter, so it sees the whole request body: the system prompt,
the user's message, and the tool results fed back on later turns. All of those
can be rewritten before they leave, and rewritten back on the way in.

THE ROUND TRIP:

    doctor  "What is Maya Cohen's eGFR?"
      |                                        (LibreChat -> guard)
      v  scrub
    model   "What is [PATIENT_7F3A2C]'s eGFR?"     <- OpenRouter sees only this
      |
      v  model emits  find_patient("[PATIENT_7F3A2C]")
      |
      v  restore                                    (guard -> LibreChat)
    tools   find_patient("Maya Cohen")              <- MCP gets the real name
      |
      v  result flows back in the next request, and is scrubbed again

LibreChat and the MCP server are inside the trust boundary and keep working with
real names; only the hop to the third-party model is de-identified. Restoring
tool-call arguments is what makes this transparent rather than merely lossy — a
scrubber that broke tool calling would be turned off within a day.

TOKENS ARE A PURE FUNCTION OF THE NAME, so no session state is needed and the
same patient gets the same token across turns and across processes. The model can
therefore still reason about "the same patient" without ever learning who it is.

THE RESIDUAL RISK, STATED PLAINLY. This scrubs identifiers it knows about —
patient names from the clinic roster. A doctor who types a date of birth, an
address, or free-text detail about a patient is not protected by this, and no
regex is going to be. It closes the specific, predictable leak this application
creates; it is not a general PHI firewall, and OpenRouter still is not a
BAA-covered processor.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# THE TOKEN MUST READ AS A NAME, NOT AN IDENTIFIER.
#
# The first design used `[PATIENT_7F3A2C]`, and a live Tier B run showed exactly
# why that was wrong. The model read it as a patient ID, skipped find_patient,
# and "normalised" it into the P### shape the tool description asks for —
# emitting get_current_biomarkers(patient_id="P084"), an ID invented from the
# token's hex digits. Two of three cases failed that way; a third passed
# `PATIENT_129CF9` through verbatim.
#
# So: no digits (nothing to build a P### from) and a name-like shape, so the
# model routes it to find_patient the way it routes any other name. The `Zx`
# prefix makes accidental matches against ordinary prose ("Patient record")
# essentially impossible while keeping it readable.
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
# The "Patient " prefix is OPTIONAL when matching, because the model does not
# always keep it. A live run showed it reading "Patient Zxsyqn" as title +
# surname and calling find_patient(name="Zxsyqn") — correct routing, but the
# bare core did not restore, so the lookup failed. The `Zx` core is distinctive
# enough to match on its own; clinical prose does not contain it by accident.
TOKEN_RE = re.compile(r"\b(?:Patient\s+)?Zx[A-Za-z]{4}\b", re.IGNORECASE)


def token_for(name: str) -> str:
    """Stable, name-shaped pseudonym.

    Derived from the name itself rather than allocated, so it survives across
    requests and processes without shared state — a conversation stays coherent
    to the model even though it is only ever seeing tokens.
    """
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).digest()
    letters = "".join(_ALPHABET[byte % 26] for byte in digest[:4])
    return f"Patient Zx{letters}"


def _token_key(text: str) -> str:
    """Normalise a matched token to its bare core, e.g. 'Patient Zxsyqn' -> 'zxsyqn'."""
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    prefix = "patient "
    return normalised[len(prefix) :] if normalised.startswith(prefix) else normalised


def _name_variants(full_name: str) -> list[str]:
    """Forms a doctor might actually type, longest first.

    Longest-first matters: replacing "Cohen" before "Maya Cohen" would leave
    "Maya [PATIENT_...]" — half-scrubbed, which is worse than either extreme
    because it looks safe.
    """
    full_name = full_name.strip()
    parts = [p for p in full_name.split() if p]
    variants = {full_name}
    if len(parts) >= 2:
        variants.add(f"{parts[-1]}, {parts[0]}")  # "Cohen, Maya"
        variants.update(parts)  # bare first or last name
    return sorted(variants, key=len, reverse=True)


@dataclass
class PhiRedactor:
    """Two-way rewriter over a known set of patient names."""

    names: tuple[str, ...] = ()
    _to_token: dict[str, str] = field(default_factory=dict, repr=False)
    _from_token: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.rebuild(self.names)

    def rebuild(self, names: tuple[str, ...]) -> None:
        self.names = tuple(names)
        self._to_token = {}
        self._from_token = {}
        for full_name in self.names:
            token = token_for(full_name)
            # Keyed on the bare core so both "Patient Zxsyqn" and "Zxsyqn"
            # resolve; see the note on TOKEN_RE.
            self._from_token[_token_key(token)] = full_name
            for variant in _name_variants(full_name):
                # A bare surname shared by two patients is ambiguous; map it to
                # whichever full name owns it first and let find_patient sort out
                # the ambiguity server-side, where it has the data to do so.
                self._to_token.setdefault(variant.lower(), token)

    @property
    def active(self) -> bool:
        return bool(self._to_token)

    def scrub(self, text: str) -> str:
        """Replace known names with tokens. Whole-word, case-insensitive."""
        if not text or not self._to_token:
            return text
        result = text
        for variant in sorted(self._to_token, key=len, reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.IGNORECASE)
            result = pattern.sub(self._to_token[variant], result)
        return result

    def restore(self, text: str) -> str:
        """Turn tokens back into names, for text re-entering the trust boundary.

        Case-insensitive and whitespace-tolerant: a model may re-type a token as
        "patient zxqvbk" or across a line break, and an unrestored token means a
        failed tool call. Unknown tokens are left alone rather than guessed at.
        """
        if not text or not self._from_token:
            return text

        def replace(match: re.Match[str]) -> str:
            return self._from_token.get(_token_key(match.group(0)), match.group(0))

        return TOKEN_RE.sub(replace, text)

    def scrub_messages(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Scrub every message in a request. Returns (messages, replacements).

        Covers user turns, assistant turns AND tool results — the tool output is
        where the real name comes back into the conversation, so scrubbing only
        the user's text would leak on the very next turn.
        """
        if not self._to_token:
            return messages, 0

        replaced = 0
        cleaned: list[dict] = []
        for message in messages:
            copy = dict(message)
            content = copy.get("content")
            if isinstance(content, str):
                scrubbed = self.scrub(content)
                if scrubbed != content:
                    replaced += len(TOKEN_RE.findall(scrubbed))
                copy["content"] = scrubbed
            elif isinstance(content, list):
                # Multimodal content blocks.
                blocks = []
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        block = {**block, "text": self.scrub(block["text"])}
                    blocks.append(block)
                copy["content"] = blocks

            if isinstance(copy.get("tool_calls"), list):
                copy["tool_calls"] = [
                    self._scrub_tool_call(call) for call in copy["tool_calls"]
                ]
            cleaned.append(copy)
        return cleaned, replaced

    def _scrub_tool_call(self, call: dict) -> dict:
        function = call.get("function")
        if not isinstance(function, dict):
            return call
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return call
        return {**call, "function": {**function, "arguments": self.scrub(arguments)}}

    def restore_tool_call(self, call: dict) -> dict:
        """Put real names back into tool arguments before LibreChat executes them.

        Without this the model would emit find_patient("[PATIENT_7F3A2C]") and the
        lookup would fail — the scrubber would have broken the product to protect
        it.
        """
        function = call.get("function")
        if not isinstance(function, dict):
            return call
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return call
        return {**call, "function": {**function, "arguments": self.restore(arguments)}}
