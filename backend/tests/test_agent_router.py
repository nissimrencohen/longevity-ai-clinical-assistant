"""The custom routing agent.

Two halves, tested differently:

* the rendering and decision logic are pure functions over tool output, so they
  are unit tested with fabricated payloads and cost nothing;
* the chain itself needs the MCP server, so those tests auto-skip when nothing is
  listening — `uv run pytest` on a clean clone still passes.

The property that matters is that the agent never invents anything: it assembles
values the tools returned and stops when a step fails, rather than continuing
with a plausible guess.
"""

from __future__ import annotations

import pytest

from agent.router import ReviewResult, _number, _preview, render, review_patient
from evals.mcp_tools import probe

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_float_noise_is_trimmed() -> None:
    """A BMI of 28.73174689021093 is false precision in a clinical note."""
    assert _number(28.73174689021093) == "28.73"
    assert _number(72.0) == "72"
    assert _number(0.5) == "0.5"
    assert _number("high") == "high"


def test_bullet_list_preview_is_readable() -> None:
    """Citing a bullet list previewed as "- **Age**.", which says nothing."""
    preview = _preview("- **Age**.\n- **Diabetes**.\n- **Hypertension**.")
    assert preview.startswith("Age. Diabetes. Hypertension.")
    assert "**" not in preview


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------


def _risk(code: str, probability: float, band: str) -> dict:
    return {
        "risk_code": code,
        "probability": probability,
        "risk_band": band,
        "time_horizon_years": 10,
        "trend_direction": "worsening",
        "drivers": [
            {
                "label": "eGFR",
                "patient_value": 52.0,
                "reference_value": 100.0,
                "direction": "increases_risk",
            }
        ],
    }


def test_only_elevated_risks_are_surfaced() -> None:
    """Grounding a low risk pads the note and invites over-reading."""
    result = ReviewResult(
        query="x",
        patient_id="P004",
        patient_name="Avraham Friedman",
        risks=[
            _risk("CKD", 0.50, "high"),
            _risk("DEMENTIA", 0.27, "intermediate"),
            _risk("CLD", 0.09, "low"),
        ],
    )
    assert [r["risk_code"] for r in result.elevated()] == ["CKD", "DEMENTIA"]


def test_all_low_is_reported_as_such() -> None:
    result = ReviewResult(
        query="x",
        patient_id="P001",
        patient_name="Maya Cohen",
        risks=[_risk("CKD", 0.02, "low")],
    )
    assert "No elevated risks" in render(result)


def test_ambiguity_stops_the_run() -> None:
    """Picking the first match is exactly the substitution error to avoid."""
    result = ReviewResult(query="Cohen", error="'Cohen' is ambiguous: Maya Cohen (P001), Ben Cohen (P009)")
    assert not result.ok
    assert "ambiguous" in render(result)
    assert "P001" in render(result)


def test_failure_is_reported_not_papered_over() -> None:
    result = ReviewResult(query="Nobody", error="no patient matching 'Nobody'")
    assert not result.ok
    rendered = render(result)
    assert "Could not complete" in rendered
    # Nothing clinical is invented on the failure path.
    assert "risk" not in rendered.lower().replace("review", "")


def test_note_carries_the_decision_support_disclaimer() -> None:
    result = ReviewResult(
        query="x", patient_id="P004", patient_name="A F", risks=[_risk("CKD", 0.5, "high")]
    )
    rendered = render(result)
    assert "Decision support only" in rendered
    assert "surrogates" in rendered
    # And never a treatment instruction — that is the guard's domain, but the
    # agent must not create the problem in the first place.
    assert "mg" not in rendered


def test_rendered_note_contains_only_tool_values() -> None:
    result = ReviewResult(
        query="x", patient_id="P004", patient_name="A F", risks=[_risk("CKD", 0.5, "high")]
    )
    rendered = render(result)
    assert "0.500" in rendered
    assert "52" in rendered and "100" in rendered


# ---------------------------------------------------------------------------
# The chain, against a live MCP server
# ---------------------------------------------------------------------------


async def _mcp_up() -> bool:
    ok, _ = await probe()
    return ok


@pytest.mark.integration
async def test_chain_resolves_and_grounds() -> None:
    if not await _mcp_up():
        pytest.skip("MCP server not reachable")

    result = await review_patient("Avraham Friedman")

    assert result.ok
    assert result.patient_id == "P004"
    assert any(s.startswith("find_patient") for s in result.steps)
    assert any(s.startswith("get_current_risks") for s in result.steps)

    ckd = next(r for r in result.risks if r["risk_code"] == "CKD")
    assert ckd["probability"] == pytest.approx(0.50, abs=1e-6)
    assert result.citations, "elevated risks should be grounded in cited guidance"


@pytest.mark.integration
async def test_chain_refuses_an_unknown_patient() -> None:
    if not await _mcp_up():
        pytest.skip("MCP server not reachable")

    result = await review_patient("Miriam Cohen")
    assert not result.ok
    assert "no patient matching" in (result.error or "")
    assert result.risks == [], "nothing should be computed for a patient that does not exist"
