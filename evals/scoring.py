"""Deterministic scorers shared by both tiers.

The interesting one is numeric faithfulness. The rule: every clinical number the
assistant states must be traceable to something a tool actually returned. An
untraceable number is a fabrication and fails the case, however plausible it
reads — a made-up lab value is the worst failure mode this system has.

Two deliberate design choices, because a scorer that cries wolf gets ignored:

* **Formatting variants are allowed.** A tool returning 0.3817 licenses "0.38",
  "0.382", "38%" and "38.2%". We compare on value, not on string.
* **Bare single digits are ignored.** "1." in a numbered list, or "5 risks", are
  structural, not clinical claims. Only decimals and multi-digit integers are
  held to the traceability rule. This trades a little recall for the precision
  that makes the metric trustworthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

BANDS = ("low", "borderline", "intermediate", "high")

# Word STEMS, matched at a word boundary (see mentions_trend). Boundary matching
# is not optional here: a bare "ris" stem for "rising" also matches "risk", which
# made every sentence containing the word "risk" score as worsening.
TREND_SYNONYMS: dict[str, tuple[str, ...]] = {
    "worsening": (
        "worsen", "worse", "increas", "rise", "rising", "rose", "climb",
        "deteriorat", "upward", "trending up", "gone up", "higher than",
    ),
    "improving": (
        "improv", "better", "decreas", "fall", "fell", "declin", "downward",
        "trending down", "gone down", "lower than",
    ),
    "stable": (
        "stable", "unchanged", "steady", "flat", "no meaningful change",
        "no significant change", "held steady",
    ),
    "insufficient_history": (
        "insufficient", "not enough history", "no prior", "only one",
    ),
}

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MATCH_TOLERANCE = 1e-6

# Numbers that are part of a UNIT rather than a measurement, and so can never be
# a fabricated clinical value. 1.73 is the body-surface-area normalisation in
# "mL/min/1.73m2" - it appears in every eGFR answer and is not a claim about the
# patient. Without this the scorer flags a correct answer, and a metric that
# cries wolf is a metric people learn to ignore.
_UNIT_NUMBERS = (1.73, 100.0)


@dataclass(frozen=True)
class SpokenNumber:
    raw: str
    value: float
    is_percent: bool


def extract_numbers(text: str) -> list[SpokenNumber]:
    """Pull candidate clinical numbers out of prose.

    Skips bare single digits (list markers, small counts) — see module docstring.
    """
    found: list[SpokenNumber] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        cleaned = raw.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue

        digits = cleaned.lstrip("-").replace(".", "")
        if "." not in cleaned and len(digits) < 2:
            continue

        tail = text[match.end() : match.end() + 1]
        found.append(SpokenNumber(raw=raw, value=value, is_percent=tail == "%"))
    return found


def _variants(value: float) -> set[float]:
    """Every rendering of one underlying quantity we accept as faithful."""
    out: set[float] = set()
    for base in (value, value * 100.0):
        out.add(base)
        for places in range(0, 5):
            out.add(round(base, places))
    return out


def collect_allowed_numbers(payloads: Iterable[Any]) -> set[float]:
    """Build the traceable-number set from tool outputs (and the user's own words).

    Walks the whole response, including numbers embedded in strings — so dates
    like "2026-01-09" license the assistant saying "January 2026" alongside 2026.
    """
    allowed: set[float] = set()
    for unit_number in _UNIT_NUMBERS:
        allowed.update(_variants(unit_number))

    def walk(node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            allowed.update(_variants(float(node)))
        elif isinstance(node, str):
            for match in _NUMBER_RE.finditer(node):
                try:
                    allowed.update(_variants(float(match.group(0).replace(",", ""))))
                except ValueError:
                    continue
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    for payload in payloads:
        walk(payload)
    return allowed


def find_untraceable_numbers(text: str, allowed: set[float]) -> list[SpokenNumber]:
    """Numbers in the prose that no tool output can account for."""
    if not allowed:
        return extract_numbers(text)

    untraceable: list[SpokenNumber] = []
    for spoken in extract_numbers(text):
        candidates = {spoken.value}
        if spoken.is_percent:
            # "38%" may be rendering a 0.38 probability.
            candidates.add(spoken.value / 100.0)
        if not any(
            any(abs(candidate - ok) <= _MATCH_TOLERANCE for ok in allowed)
            for candidate in candidates
        ):
            untraceable.append(spoken)
    return untraceable


def mentions_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def mentions_band(text: str, band: str) -> bool:
    """Did the assistant use this band word?

    'high' also matches the phrase 'high risk'; word boundaries keep it from
    matching 'higher', which is a comparison rather than a band.
    """
    return mentions_word(text, band)


def contradicting_bands(text: str, expected: str) -> list[str]:
    """Other band words present — a weak signal the assistant hedged or mislabelled."""
    return [b for b in BANDS if b != expected and mentions_band(text, b)]


def mentions_trend(text: str, direction: str) -> bool:
    """Does the prose express this direction of travel?

    Stems are anchored at a word boundary so "rise" matches "rising" but not
    "risk", and "fall" matches "falling" but not "pitfall".
    """
    for stem in TREND_SYNONYMS.get(direction, (direction,)):
        if re.search(rf"\b{re.escape(stem)}", text, re.IGNORECASE):
            return True
    return False


def find_value(payload: Any, *path: str) -> Any:
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def risk_entry(payload: Any, risk_code: str) -> dict[str, Any] | None:
    risks = find_value(payload, "risks")
    if not isinstance(risks, list):
        return None
    for risk in risks:
        if isinstance(risk, dict) and risk.get("risk_code") == risk_code:
            return risk
    return None


def trend_series(payload: Any, risk_code: str) -> list[dict[str, Any]]:
    trends = find_value(payload, "trends")
    if isinstance(trends, dict):
        series = trends.get(risk_code)
        if isinstance(series, list):
            return series
    return []


def numbers_close(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance + 1e-12
