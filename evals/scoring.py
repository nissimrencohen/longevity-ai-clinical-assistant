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

# A leading '-' counts as a minus sign only when it does NOT follow a digit or a
# dot. Otherwise "normal range 70-99" parses as 70 and MINUS 99, and "2026-06-15"
# as 2026, -6, -15 — turning ordinary ranges and dates into phantom values.
_NUMBER_RE = re.compile(r"(?<![\d.])-?\d[\d,]*(?:\.\d+)?")
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
    # Offset in the source text. Carried so callers can read the words around a
    # number; locating it later with text.find() would return the FIRST
    # occurrence, which is the wrong one whenever a value repeats.
    start: int = -1


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
        found.append(
            SpokenNumber(
                raw=raw, value=value, is_percent=tail == "%", start=match.start()
            )
        )
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

    # Probabilities seen anywhere in the payloads, kept so their pairwise
    # DIFFERENCES can be allowed too — see the note at the end of this function.
    probabilities: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            allowed.update(_variants(float(node)))
            if 0.0 < float(node) < 1.0:
                probabilities.add(float(node))
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

    # Comparing two risks is the natural derived operation in this domain, and
    # the arithmetic is not a fabrication: "0.450 vs 0.393, about 5.7 percentage
    # points higher" is three traceable numbers, the third being the difference
    # of the first two. Without this the scorer failed a correct comparison.
    #
    # Deliberately narrow — only differences between values in (0, 1), i.e.
    # probabilities. Allowing arbitrary pairwise differences over every number in
    # the payload would widen the traceable set enough that a genuinely invented
    # lab value could land in it by coincidence, which is the whole thing this
    # check exists to catch.
    ordered = sorted(probabilities)
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            allowed.update(_variants(round(second - first, 10)))

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


# Cues that mark a number as a GUIDELINE THRESHOLD rather than a claim about this
# patient: "ideally >60", "target <100", "normal range 70-99".
_REFERENCE_CUE_RE = re.compile(
    r"(?:[<>]=?|[≤≥]|ideal(?:ly)?|target|goal|optimal|threshold|"
    r"normal|reference|range|above|below|over|under|at least|no more than|"
    r"less than|greater than|higher than|lower than)\s*[~about]*\s*$",
    re.IGNORECASE,
)


def split_untraceable(
    text: str, allowed: set[float]
) -> tuple[list[SpokenNumber], list[SpokenNumber]]:
    """Separate untraceable numbers into patient claims vs guideline references.

    These are different failures and conflating them corrupts the headline metric.

        "her eGFR is 87"          -> a fabricated patient value. Critical.
        "HDL 52 (ideally >60)"    -> an unverifiable guideline threshold recalled
                                     from training. Worth surfacing, but not the
                                     same thing as inventing a lab result.

    Detection is by the text immediately preceding the number: a comparator or a
    reference word ("ideally", "target", "normal range") means the number is
    being quoted as a threshold, not asserted as this patient's measurement.

    Returns ``(patient_values, reference_values)``.
    """
    patient: list[SpokenNumber] = []
    reference: list[SpokenNumber] = []

    for spoken in find_untraceable_numbers(text, allowed):
        index = spoken.start
        preceding = text[max(0, index - 28) : index] if index >= 0 else ""
        if _has_reference_cue(preceding):
            reference.append(spoken)
        else:
            patient.append(spoken)
    return patient, reference


# "130/80" is one blood-pressure expression, so a cue before the pair governs
# both halves. Without this, "target ~130/80" reads 130 as a threshold and 80 as
# a fabricated patient value.
_PAIRED_VALUE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*/\s*$")


def _has_reference_cue(preceding: str) -> bool:
    if _REFERENCE_CUE_RE.search(preceding):
        return True
    stripped = _PAIRED_VALUE_RE.sub("", preceding)
    return stripped != preceding and _REFERENCE_CUE_RE.search(stripped) is not None


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
