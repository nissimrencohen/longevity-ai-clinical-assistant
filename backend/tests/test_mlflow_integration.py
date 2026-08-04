"""Integration tests against a LIVE MLflow model server.

The rest of the suite mocks MLflow at the HTTP layer so it runs in milliseconds
with no services up. That proves the maths and the wiring inside our process — but
it cannot prove the router is registered correctly, that the ``model`` param
routes, or that the server returns probabilities rather than class labels.

This file covers exactly that seam, and skips (rather than fails) when nothing is
listening on :5001, so ``make test`` stays green on a clean checkout.

    uv run mlflow models serve -m models/mlflow_risk_router -p 5001 --env-manager local
    uv run pytest backend/tests/test_mlflow_integration.py
"""

from __future__ import annotations

import httpx
import pytest

from backend.app.core.config import settings
from backend.app.services.mlflow_client import MLflowRiskClient
from backend.app.services.risk import RiskService

pytestmark = pytest.mark.integration


def _server_is_up() -> bool:
    try:
        httpx.post(
            settings.mlflow_url,
            json={
                "dataframe_split": {"columns": ["age_years"], "data": [[50]]},
                "params": {"model": "framingham_ckd"},
            },
            timeout=2.0,
        )
    except httpx.HTTPError:
        return False
    return True


requires_server = pytest.mark.skipif(
    not _server_is_up(),
    reason="MLflow model server not reachable on :5001 (see GUIDE.md §4)",
)


@requires_server
async def test_guide_smoke_payload_returns_the_anchor_probability() -> None:
    """The exact curl from GUIDE.md §4 — P004's CKD payload, expected ~0.50."""
    async with httpx.AsyncClient(timeout=10.0) as http:
        client = MLflowRiskClient(http, settings.mlflow_url)
        probability = await client.predict_proba(
            "framingham_ckd",
            {
                "age_years": 72,
                "diabetes": 1,
                "hypertension": 1,
                "proteinuria_trace_plus": 1,
                "egfr": 52,
            },
        )
    assert probability == pytest.approx(0.50, abs=1e-6)


@requires_server
async def test_all_five_models_route_and_return_probabilities() -> None:
    """Every model must be reachable by name and return a value in (0, 1).

    A value of exactly 0.0 or 1.0 across the board would mean the router is
    returning class labels — the failure mode GUIDE trap #5 warns about.
    """
    async with httpx.AsyncClient(timeout=10.0) as http:
        service = RiskService(MLflowRiskClient(http, settings.mlflow_url))
        response = await service.get_current_risks("P004")

    assert {r.risk_code for r in response.risks} == {
        "CVD", "T2DM", "CKD", "CLD", "DEMENTIA"
    }
    for risk in response.risks:
        assert 0.0 < risk.probability < 1.0, f"{risk.risk_code} looks like a label"


@requires_server
async def test_live_and_mocked_paths_agree(risk_service) -> None:
    """The mocked transport must reproduce the live server exactly.

    This is what licenses the rest of the suite to mock: if these two ever diverge,
    the fast tests have stopped meaning anything.
    """
    mocked = await risk_service.get_current_risks("P006")

    async with httpx.AsyncClient(timeout=10.0) as http:
        live_service = RiskService(MLflowRiskClient(http, settings.mlflow_url))
        live = await live_service.get_current_risks("P006")

    mocked_by_code = {r.risk_code: r.probability for r in mocked.risks}
    for risk in live.risks:
        assert risk.probability == pytest.approx(mocked_by_code[risk.risk_code], abs=1e-9)


@requires_server
async def test_unknown_model_name_is_rejected() -> None:
    """The router should refuse an unrecognised model rather than guessing one."""
    async with httpx.AsyncClient(timeout=10.0) as http:
        response = await http.post(
            settings.mlflow_url,
            json={
                "dataframe_split": {"columns": ["age_years"], "data": [[50]]},
                "params": {"model": "not_a_real_model"},
            },
        )
    assert response.status_code >= 400
