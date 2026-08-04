"""Acceptance tests for the clinical API.

These are the four tests the assignment ships as its acceptance spec, with the
``@pytest.mark.skip`` decorators removed. Their bodies are unchanged.

The model server is mocked at the HTTP layer (see ``conftest.py``) but computes
from the real pickles, so the probabilities asserted here are the same numbers a
live MLflow router returns. ``test_mlflow_integration.py`` covers the live wiring.

Run:  uv run pytest
"""

from __future__ import annotations

from httpx import AsyncClient

RISK_CODES = {"CVD", "T2DM", "CKD", "CLD", "DEMENTIA"}
BANDS = {"low", "borderline", "intermediate", "high"}


async def test_health(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_biomarkers_known_patient(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "P001"})
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "P001"
    # Maya Cohen's eGFR in the mock data is 102.
    assert body["biomarkers"]["egfr_ml_min_1_73m2"] == 102


async def test_unknown_patient_returns_404(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "NOPE"})
    assert r.status_code == 404


async def test_risks_returns_five_bands(client: AsyncClient) -> None:
    r = await client.get("/api/v1/get_current_risks", params={"patient_id": "P004"})
    assert r.status_code == 200
    risks = r.json()["risks"]
    assert {x["risk_code"] for x in risks} == RISK_CODES
    for x in risks:
        assert 0.0 <= x["probability"] <= 1.0
        assert x["risk_band"] in BANDS
    # P004 (Avraham Friedman) is the designed CKD patient — should read high.
    ckd = next(x for x in risks if x["risk_code"] == "CKD")
    assert ckd["risk_band"] == "high"


async def test_risks_are_appended(client: AsyncClient) -> None:
    """Calling get_current_risks should persist today's values to the risks log."""
    import sqlite3

    from backend.app.core.config import settings

    def row_count() -> int:
        con = sqlite3.connect(settings.patient_db_path)
        try:
            return con.execute(
                "SELECT COUNT(*) FROM risks WHERE patient_id='P002' AND computed_at >= date('now')"
            ).fetchone()[0]
        finally:
            con.close()

    await client.get("/api/v1/get_current_risks", params={"patient_id": "P002"})
    assert row_count() >= 5


# ---------------------------------------------------------------------------
# Additional endpoint-level cases (mine).
# ---------------------------------------------------------------------------


async def test_unknown_patient_risks_returns_404(client: AsyncClient) -> None:
    """The safety-critical path: P999 must 404, never a fabricated risk panel.

    Mirrors the `safety-unknown-p999` eval case — if this leaks a 200 with an
    empty risk list, the assistant has a template to hallucinate into.
    """
    r = await client.get("/api/v1/get_current_risks", params={"patient_id": "P999"})
    assert r.status_code == 404
    assert "P999" in r.json()["detail"]


async def test_model_server_down_returns_502(client_no_model_server: AsyncClient) -> None:
    """An unreachable model server must surface as 502, not a default probability."""
    r = await client_no_model_server.get(
        "/api/v1/get_current_risks", params={"patient_id": "P001"}
    )
    assert r.status_code == 502


async def test_biomarkers_reports_age_at_clinic_today(client: AsyncClient) -> None:
    """Age is derived against the fixed clinic date, not the wall clock."""
    r = await client.get("/api/v1/get_current_biomarkers", params={"patient_id": "P004"})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Avraham Friedman"
    assert body["age_years"] == 72  # 1954-02-18 at 2026-07-09
    assert body["sex"] == "male"


async def test_every_patient_scores_all_five_risks(client: AsyncClient) -> None:
    """All eight patients score without error — the NULL-input regression, end to end.

    Before `gestational_diabetes` was coalesced, this failed for exactly the four
    male patients (NaN into sklearn).
    """
    for pid in ("P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008"):
        r = await client.get("/api/v1/get_current_risks", params={"patient_id": pid})
        assert r.status_code == 200, f"{pid} failed: {r.text}"
        risks = r.json()["risks"]
        assert {x["risk_code"] for x in risks} == RISK_CODES
        assert all(0.0 <= x["probability"] <= 1.0 for x in risks)
