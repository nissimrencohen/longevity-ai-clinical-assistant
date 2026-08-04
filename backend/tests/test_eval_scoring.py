"""Tests for the eval harness's scorers.

The scorers decide whether the assistant fabricated a number, so they need to be
correct in both directions. A scorer that misses fabrications is useless; one
that flags correct answers gets ignored, which is worse — it trains you to
dismiss real failures.

(Lives under backend/tests so `uv run pytest` picks it up with everything else.)
"""

from __future__ import annotations

import pytest

from evals.scoring import (
    collect_allowed_numbers,
    contradicting_bands,
    extract_numbers,
    find_untraceable_numbers,
    mentions_band,
    mentions_trend,
)

TOOL_PAYLOAD = {
    "patient_id": "P001",
    "name": "Maya Cohen",
    "biomarkers": {
        "measured_at": "2026-06-15",
        "egfr_ml_min_1_73m2": 102.0,
        "hdl_cholesterol_mgdl": 68.0,
    },
    "risks": [
        {"risk_code": "CKD", "probability": 0.5, "risk_band": "high",
         "time_horizon_years": 10, "trend_direction": "worsening"}
    ],
}


def _untraceable(text: str, payloads=(TOOL_PAYLOAD,)) -> list[str]:
    return [n.raw for n in find_untraceable_numbers(text, collect_allowed_numbers(payloads))]


# --- numbers that must be accepted ------------------------------------------


def test_exact_value_is_traceable() -> None:
    assert _untraceable("Her eGFR is 102.0 mL/min.") == []


def test_rounded_value_is_traceable() -> None:
    """0.5 may be spoken as 0.50, 0.500, or 50%."""
    for rendering in ("0.5", "0.50", "0.500", "50%", "50.0%"):
        assert _untraceable(f"CKD risk is {rendering}.") == [], rendering


def test_egfr_unit_number_is_not_a_fabrication() -> None:
    """1.73 belongs to the unit mL/min/1.73m2, not to the patient.

    Regression: the first Tier B run failed a correct answer over this.
    """
    assert _untraceable("eGFR is 102.0 mL/min/1.73m2 (measured 2026-06-15).") == []


def test_dates_from_the_payload_are_traceable() -> None:
    assert _untraceable("Measured on 2026-06-15.") == []


def test_single_digits_are_ignored_as_structural() -> None:
    """List markers and small counts are not clinical claims."""
    assert _untraceable("1. CKD 2. CVD - across 5 risks.") == []


# --- numbers that must be caught ---------------------------------------------


def test_invented_lab_value_is_caught() -> None:
    assert _untraceable("Her eGFR is 87.4 mL/min.") == ["87.4"]


def test_invented_probability_is_caught() -> None:
    assert _untraceable("CKD risk is 0.62.") == ["0.62"]


def test_plausible_but_unsourced_number_is_caught() -> None:
    """The worst failure mode: a confident number nobody measured."""
    assert _untraceable("Her HbA1c is 6.4% and LDL is 143 mg/dL.") == ["6.4", "143"]


def test_everything_is_untraceable_when_no_tool_ran() -> None:
    """A tool error means no allowed set, so any stated value is a fabrication."""
    assert _untraceable("eGFR is 102.0", payloads=()) == ["102.0"]


# --- extraction, bands, trends -----------------------------------------------


def test_extract_skips_bare_single_digits() -> None:
    assert [n.raw for n in extract_numbers("1 patient, 12 labs, 0.5 risk")] == ["12", "0.5"]


def test_percent_is_flagged() -> None:
    assert extract_numbers("38%")[0].is_percent is True


@pytest.mark.parametrize("band", ["low", "borderline", "intermediate", "high"])
def test_band_word_detected(band: str) -> None:
    assert mentions_band(f"The risk is {band} for this patient.", band)


def test_higher_does_not_match_the_high_band() -> None:
    """'higher' is a comparison, not a band label."""
    assert not mentions_band("Rivka's risk is higher than David's.", "high")


def test_contradicting_bands_reported() -> None:
    assert contradicting_bands("It is high, not low.", "high") == ["low"]


@pytest.mark.parametrize(
    ("text", "direction"),
    [
        ("the risk has worsened over time", "worsening"),
        ("kidney risk is increasing", "worsening"),
        ("it has improved since January", "improving"),
        ("the value is decreasing", "improving"),
        ("the risk has remained stable", "stable"),
        ("essentially unchanged", "stable"),
    ],
)
def test_trend_synonyms(text: str, direction: str) -> None:
    assert mentions_trend(text, direction)


def test_trend_direction_not_confused() -> None:
    assert not mentions_trend("the risk has improved", "worsening")
