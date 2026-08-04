"""Service-level tests: the anchor oracle, the dedupe rule, and trends."""

from __future__ import annotations

import sqlite3

import pytest

from backend.app.core.config import settings
from backend.app.core.errors import ModelServerError, PatientNotFoundError


def _risk(response, code: str):
    return next(r for r in response.risks if r.risk_code == code)


def _count_rows(patient_id: str, risk_code: str | None = None) -> int:
    con = sqlite3.connect(settings.patient_db_path)
    try:
        if risk_code:
            return con.execute(
                "SELECT COUNT(*) FROM risks WHERE patient_id=? AND risk_code=?",
                (patient_id, risk_code),
            ).fetchone()[0]
        return con.execute(
            "SELECT COUNT(*) FROM risks WHERE patient_id=?", (patient_id,)
        ).fetchone()[0]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# The anchor oracle.
# ---------------------------------------------------------------------------


async def test_p004_ckd_hits_the_calibration_anchor(risk_service) -> None:
    """P004's CKD payload IS the model's high-risk calibration anchor.

    models/generate_models.py calibrates framingham_ckd so that
    (age 72, diabetes 1, hypertension 1, proteinuria 1, egfr 52) -> p_high = 0.50.
    P004's record derives exactly that vector, so 0.50 is an exact expected value,
    not a tolerance. If the feature mapping, units, or ordering ever drift, this
    is the test that catches it.
    """
    response = await risk_service.get_current_risks("P004")
    ckd = _risk(response, "CKD")
    assert ckd.probability == pytest.approx(0.50, abs=1e-6)
    assert ckd.risk_band == "high"
    assert ckd.time_horizon_years == 10
    assert ckd.model_name == "framingham_ckd"


async def test_designed_headline_risks_match_the_data_dictionary(risk_service) -> None:
    """Each patient's designed headline risk should read high (sanity reference table)."""
    expected = {
        "P002": "CVD",
        "P003": "T2DM",
        "P004": "CKD",
        "P005": "CLD",
        "P006": "DEMENTIA",
    }
    for patient_id, risk_code in expected.items():
        response = await risk_service.get_current_risks(patient_id)
        assert _risk(response, risk_code).risk_band == "high", (
            f"{patient_id} {risk_code} was {_risk(response, risk_code).probability:.3f}"
        )


async def test_healthy_patient_has_no_elevated_risks(risk_service) -> None:
    """P001 is the designed healthy patient — nothing should read elevated."""
    response = await risk_service.get_current_risks("P001")
    assert all(r.risk_band == "low" for r in response.risks), [
        (r.risk_code, r.probability) for r in response.risks
    ]


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


async def test_unknown_patient_raises(risk_service) -> None:
    with pytest.raises(PatientNotFoundError):
        await risk_service.get_current_risks("P999")


async def test_model_server_down_raises(risk_service_no_model_server) -> None:
    with pytest.raises(ModelServerError):
        await risk_service_no_model_server.get_current_risks("P001")


async def test_non_probability_response_is_rejected(risk_service_returns_labels) -> None:
    """Guards the GUIDE's trap #5: a router calling .predict() returns 0/1 labels.

    A label of 1.0 is indistinguishable from a probability of 1.0, so we cannot
    catch every case — but anything outside [0, 1] must never reach a clinician.
    """
    with pytest.raises(ModelServerError):
        await risk_service_returns_labels.get_current_risks("P001")


# ---------------------------------------------------------------------------
# The append / dedupe rule — this is the "GET that writes" answer.
# ---------------------------------------------------------------------------


async def test_first_call_appends_one_row_per_risk(risk_service) -> None:
    before = _count_rows("P002")
    response = await risk_service.get_current_risks("P002")
    after = _count_rows("P002")

    assert after == before + 5
    assert all(r.persisted for r in response.risks)


async def test_repeat_call_is_idempotent(risk_service) -> None:
    """Unchanged inputs must not spam the trend — and this is what makes the GET safe.

    The probabilities are still recomputed and returned; only the append is
    skipped, so a doctor refreshing the page produces no observable state change.
    """
    first = await risk_service.get_current_risks("P002")
    after_first = _count_rows("P002")

    second = await risk_service.get_current_risks("P002")
    after_second = _count_rows("P002")

    assert after_second == after_first, "a second identical call must not append"
    assert all(not r.persisted for r in second.risks)
    # The numbers are unchanged, because the model is deterministic.
    for a, b in zip(first.risks, second.risks, strict=True):
        assert a.probability == pytest.approx(b.probability)


async def test_changed_inputs_append_a_new_row(risk_service) -> None:
    """When a biomarker actually changes, the trend must gain a point."""
    await risk_service.get_current_risks("P002")
    before = _count_rows("P002", "CKD")

    con = sqlite3.connect(settings.patient_db_path)
    try:
        con.execute(
            "UPDATE biomarkers SET egfr_ml_min_1_73m2 = 45 WHERE patient_id = 'P002'"
        )
        con.commit()
    finally:
        con.close()

    response = await risk_service.get_current_risks("P002")
    assert _count_rows("P002", "CKD") == before + 1
    assert _risk(response, "CKD").persisted


async def test_inputs_json_is_written_for_audit(risk_service) -> None:
    """Every appended row must carry the exact payload that produced it."""
    import json

    await risk_service.get_current_risks("P004")
    con = sqlite3.connect(settings.patient_db_path)
    try:
        raw = con.execute(
            "SELECT inputs_json FROM risks WHERE patient_id='P004' AND risk_code='CKD' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()

    audit = json.loads(raw)
    assert audit["model_name"] == "framingham_ckd"
    assert audit["clinic_today"] == "2026-07-09"
    assert audit["features"] == {
        "age_years": 72.0,
        "diabetes": 1.0,
        "hypertension": 1.0,
        "proteinuria_trace_plus": 1.0,
        "egfr": 52.0,
    }
    assert len(audit["inputs_hash"]) == 64


async def test_audit_records_the_gestational_diabetes_default(risk_service) -> None:
    """The one sanctioned imputation must be traceable, per male patient."""
    import json

    await risk_service.get_current_risks("P004")
    con = sqlite3.connect(settings.patient_db_path)
    try:
        raw = con.execute(
            "SELECT inputs_json FROM risks WHERE patient_id='P004' AND risk_code='T2DM' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()

    assert json.loads(raw)["defaults_applied"] == {"gestational_diabetes": 0.0}


# ---------------------------------------------------------------------------
# Trends.
# ---------------------------------------------------------------------------


async def test_ckd_trend_for_p004_is_worsening(risk_service) -> None:
    """The `trend-ckd-p004` eval case: seeded 0.39 -> 0.45, live 0.50."""
    response = await risk_service.get_current_risks("P004")

    points = response.trends["CKD"]
    assert [round(p.probability, 3) for p in points[-3:]] == [0.39, 0.45, 0.5]
    assert _risk(response, "CKD").trend_direction == "worsening"


async def test_trend_points_are_chronological(risk_service) -> None:
    response = await risk_service.get_current_risks("P004")
    for points in response.trends.values():
        timestamps = [p.computed_at for p in points]
        assert timestamps == sorted(timestamps)
