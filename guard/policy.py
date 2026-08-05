"""Clinical safety policy: detect and remove prescribing instructions.

WHY THIS EXISTS. The assistant was told, in three separate places, never to
recommend starting a medication. It did anyway — 4 of 7 observed runs issued a
definitive recommendation for "atorvastatin 40 mg daily", and strengthening the
wording made the rate worse, not better. Prompts are advisory; this is not.

WHAT COUNTS AS A PRESCRIBING INSTRUCTION. Not every mention of a drug. The
assistant legitimately reports that a patient is *on* a statin (it is a model
input), and legitimately says a decision is worth discussing. What it must not do
is issue the instruction itself. Three signals, any of which fires:

  A. A written prescription      "Atorvastatin 40 mg daily."
  B. A directive to start        "Start him on atorvastatin."
  C. An authoritative endorsement "40 mg daily is a reasonable starting dose."

HEDGES ARE ALLOWED, DELIBERATELY. "Consider whether statin therapy is
appropriate" and "this is worth discussing with him" are exactly the register a
decision-support tool should use, and blocking them would push the assistant
toward saying nothing useful. A guard that suppresses good clinical
communication gets switched off. So a hedge in the same sentence downgrades a
directive to advice — UNLESS the sentence also writes out a dose, because
"consider atorvastatin 40 mg daily" is still a prescription with a softener in
front of it.

There is no formulary here, so drugs are recognised by a small lexicon plus
common stem suffixes (-statin, -pril, -sartan, -olol, -formin, ...). That is a
heuristic and it will miss unusual agents; the dose+frequency rule catches most
of what the lexicon does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- vocabulary -------------------------------------------------------------

_DRUG_SUFFIXES = (
    "statin", "pril", "sartan", "olol", "formin", "gliptin", "glifozin",
    "gliflozin", "prazole", "parin", "warfarin", "floxacin", "cillin",
    "mycin", "azide", "semide", "dipine", "tinib", "mab", "zepam", "codone",
)
_DRUG_WORDS = (
    "aspirin", "metformin", "insulin", "statin", "statins", "ezetimibe",
    "clopidogrel", "warfarin", "heparin", "allopurinol", "levothyroxine",
    "amlodipine", "atorvastatin", "rosuvastatin", "simvastatin", "ramipril",
    "lisinopril", "losartan", "bisoprolol", "furosemide", "empagliflozin",
    "semaglutide", "liraglutide", "sglt2", "glp-1", "ace inhibitor",
    "beta blocker", "anticoagulant", "antihypertensive",
)

_DRUG_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(w) for w in _DRUG_WORDS)
    + r"|\w*(?:"
    + "|".join(_DRUG_SUFFIXES)
    + r"))\b",
    re.IGNORECASE,
)

_DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|µg|g|units?|iu|ml)\b", re.IGNORECASE
)
_FREQUENCY_RE = re.compile(
    r"\b(?:once|twice|three times|four times)?\s*(?:daily|nightly|weekly|"
    r"per day|a day|at night|in the morning|bd|bid|tid|qid|qd|od|prn)\b",
    re.IGNORECASE,
)

# Directives: the assistant telling someone to act.
_DIRECTIVE_RE = re.compile(
    r"\b(?:start(?:ing)?|initiat(?:e|ing)|begin(?:ning)?|commenc(?:e|ing)|"
    r"prescrib(?:e|ing)|put\s+(?:him|her|them)\s+on|switch(?:ing)?\s+(?:him|her|them)?\s*to|"
    r"titrat(?:e|ing)|increase\s+the\s+dose|add)\b",
    re.IGNORECASE,
)

# Authoritative endorsement: not an imperative, but still a decision.
_ENDORSEMENT_RE = re.compile(
    r"\b(?:is|would be|seems|appears)\s+(?:a\s+|an\s+)?"
    r"(?:reasonable|appropriate|suitable|sensible|indicated|warranted|justified|"
    r"advisable|the\s+right)\b"
    r"|\bi\s+(?:recommend|advise|suggest)\b"
    r"|\bshould\s+(?:be\s+)?(?:start|begin|initiat|prescrib|receiv|take|be\s+put)\w*",
    re.IGNORECASE,
)

# Language that makes a statement advisory rather than instructive.
_HEDGE_RE = re.compile(
    r"\b(?:consider|considering|discuss(?:ing)?|whether|may\s+(?:want|wish)|"
    r"might\s+(?:want|wish|be)|could\s+be|worth\s+(?:discussing|considering)|"
    r"evaluat\w*|assess\w*|weigh\w*|your\s+(?:call|judgement|judgment|decision)|"
    r"clinician'?s?\s+(?:judgement|judgment|decision)|defer)\b",
    re.IGNORECASE,
)

NOTICE = (
    "\n\n---\n"
    "*A prescribing recommendation was removed by the clinical safety guard. "
    "This assistant provides decision support only — medication decisions rest "
    "with the treating clinician.*"
)
REDACTION_NOTICE = (
    "\n\n---\n"
    "*A specific medication and dose were redacted by the clinical safety guard. "
    "This assistant provides decision support only — medication decisions rest "
    "with the treating clinician.*"
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class GuardVerdict:
    """What the guard decided, and why. Recorded for audit."""

    text: str
    triggered: bool = False
    rules: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def to_audit(self) -> dict:
        return {
            "triggered": self.triggered,
            "rules": self.rules,
            "removed_count": len(self.removed),
            # The offending text is kept so a reviewer can judge the guard
            # itself; it never reaches the clinician.
            "removed": self.removed,
        }


# A specific agent with a specific dose: "atorvastatin 40 mg daily". Naming one
# is the prescriber's call, so it is redacted even inside an otherwise properly
# deferential sentence — but redacted, not deleted, so the reasoning survives.
_SPECIFIC_RX_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(w) for w in _DRUG_WORDS)
    + r"|\w*(?:"
    + "|".join(_DRUG_SUFFIXES)
    + r"))\s*"
    r"(?:\d+(?:\.\d+)?\s*(?:mg|mcg|ug|µg|g|units?|iu))"
    r"(?:\s*(?:once|twice|three times)?\s*"
    r"(?:daily|nightly|weekly|per day|a day|at night|bd|bid|tid|qid|qd|od))?",
    re.IGNORECASE,
)
_REDACTION = "this medication"


def classify_sentence(sentence: str) -> str | None:
    """Return the rule a sentence violates, or None if it is acceptable.

    ``hedged_specific_dose`` is a softer verdict: the sentence defers correctly
    but still names an agent and dose, so the prescription is redacted and the
    rest of the sentence kept.
    """
    if not _DRUG_RE.search(sentence):
        return None

    has_dose = bool(_DOSE_RE.search(sentence))
    has_frequency = bool(_FREQUENCY_RE.search(sentence))
    has_directive = bool(_DIRECTIVE_RE.search(sentence))
    has_endorsement = bool(_ENDORSEMENT_RE.search(sentence))
    hedged = bool(_HEDGE_RE.search(sentence))

    if has_dose and (has_frequency or has_directive or has_endorsement):
        # Deleting the whole sentence here was too blunt. A real guarded run
        # removed "Whether to initiate atorvastatin 40 mg daily depends on your
        # assessment of his risk, his preferences, any contraindications..." —
        # which hands the decision to the clinician exactly as it should, and
        # whose only fault is naming the agent and dose. Redact the
        # prescription, keep the reasoning.
        return "hedged_specific_dose" if hedged else "written_prescription"

    if hedged:
        # "Consider whether a statin is appropriate" — advisory, and exactly the
        # register we want a decision-support tool to use.
        return None

    if has_directive:
        return "directive_to_start"
    if has_endorsement:
        return "authoritative_endorsement"
    return None


def redact_prescription(sentence: str) -> str:
    """Strip the specific agent+dose, leaving the clinical reasoning intact."""
    return _SPECIFIC_RX_RE.sub(_REDACTION, sentence)


def enforce(text: str) -> GuardVerdict:
    """Strip prescribing instructions from an assistant message.

    Sentence-level rather than all-or-nothing: the surrounding clinical summary
    is usually correct and useful, and discarding it would make the guard
    expensive enough that someone disables it.
    """
    if not text or not text.strip():
        return GuardVerdict(text=text)

    kept: list[str] = []
    removed: list[str] = []
    rules: list[str] = []

    redacted_any = False
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        if not sentence.strip():
            continue
        rule = classify_sentence(sentence)
        if rule == "hedged_specific_dose":
            # Properly deferential, but names an agent and dose. Keep the
            # sentence, drop the prescription.
            kept.append(redact_prescription(sentence))
            removed.append(sentence.strip())
            redacted_any = True
            if rule not in rules:
                rules.append(rule)
        elif rule:
            removed.append(sentence.strip())
            if rule not in rules:
                rules.append(rule)
        else:
            kept.append(sentence)

    if not removed:
        return GuardVerdict(text=text)

    cleaned = " ".join(s.strip() for s in kept).strip()
    if not cleaned:
        cleaned = (
            "I can't provide a prescribing recommendation. I can summarise this "
            "patient's risk profile and the factors driving it if that would help."
        )

    # Only a redaction happened — say so accurately rather than claiming a
    # recommendation was removed.
    only_redacted = redacted_any and rules == ["hedged_specific_dose"]
    notice = REDACTION_NOTICE if only_redacted else NOTICE
    return GuardVerdict(
        text=cleaned + notice, triggered=True, rules=rules, removed=removed
    )
