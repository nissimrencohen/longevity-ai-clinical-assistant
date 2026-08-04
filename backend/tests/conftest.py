"""Shared pytest fixtures for the backend tests.

Two decisions worth explaining, because they are what make this suite both fast
and trustworthy:

1. **The database is a per-test temp copy.** The shipped ``data/patient_db.db`` is
   the gold fixture the eval cases are anchored to; a test suite that appends risk
   rows to it would quietly corrupt that. ``settings.patient_db_path`` is patched
   to a throwaway copy, which also means the provided ``test_risks_are_appended``
   (which opens ``settings.patient_db_path`` directly with stdlib sqlite3) keeps
   working unmodified.

2. **MLflow is mocked at the HTTP layer, but the numbers are real.** The mock
   transport loads the actual ``models/*.pkl`` and runs ``predict_proba`` exactly
   as the router does. So the tests need no running server, yet still assert
   genuine probabilities — a mock that returned 0.42 for everything would make
   "numeric faithfulness" tests meaningless.
"""

from __future__ import annotations

import json
import pickle
import shutil
from functools import lru_cache
from pathlib import Path

import httpx
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app.api.deps import get_risk_service
from backend.app.core.config import settings
from backend.app.main import app
from backend.app.services.mlflow_client import MLflowRiskClient
from backend.app.services.risk import RiskService

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
GOLDEN_DB = REPO_ROOT / "data" / "patient_db.db"

MODEL_FILES = {
    "prevent_cvd": "prevent_cvd.pkl",
    "ada_t2dm": "ada_t2dm.pkl",
    "framingham_ckd": "framingham_ckd.pkl",
    "clivd_cld": "clivd_cld.pkl",
    "caide_dementia": "caide_dementia.pkl",
}


@lru_cache(maxsize=None)
def load_model(name: str):
    """Load a pickled model once per session."""
    with (MODELS_DIR / MODEL_FILES[name]).open("rb") as fh:
        return pickle.load(fh)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch) -> Path:
    """Point the app at a disposable copy of the patient database."""
    target = tmp_path / "patient_db.db"
    shutil.copyfile(GOLDEN_DB, target)
    monkeypatch.setattr(settings, "patient_db_path", target)
    return target


def _router_handler(request: httpx.Request) -> httpx.Response:
    """Stand-in for the MLflow RiskRouter — same contract, same maths."""
    body = json.loads(request.content)
    model_name = body["params"]["model"]
    split = body["dataframe_split"]

    model = load_model(model_name)
    frame = pd.DataFrame(split["data"], columns=split["columns"])
    missing = [c for c in model.feature_names_in_ if c not in frame.columns]
    if missing:
        return httpx.Response(400, json={"error": f"missing features: {missing}"})

    X = frame[list(model.feature_names_in_)].astype("float64")
    return httpx.Response(200, json={"predictions": model.predict_proba(X)[:, 1].tolist()})


@pytest.fixture
def mlflow_transport() -> httpx.MockTransport:
    return httpx.MockTransport(_router_handler)


@pytest.fixture
def unreachable_transport() -> httpx.MockTransport:
    """Simulates the model server being down (GUIDE trap: must surface as 502)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.MockTransport(handler)


@pytest.fixture
def label_transport() -> httpx.MockTransport:
    """A router wrongly wired to sklearn `.predict()` — returns class labels.

    GUIDE trap #5. Labels come back as 0/1, so we emit an out-of-range value to
    assert the client refuses anything that is not a probability.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"predictions": [1.7]})

    return httpx.MockTransport(handler)


@pytest_asyncio.fixture
async def risk_service(mlflow_transport) -> RiskService:
    async with httpx.AsyncClient(transport=mlflow_transport) as http:
        yield RiskService(MLflowRiskClient(http, "http://model-server/invocations"))


@pytest_asyncio.fixture
async def risk_service_no_model_server(unreachable_transport) -> RiskService:
    async with httpx.AsyncClient(transport=unreachable_transport) as http:
        yield RiskService(MLflowRiskClient(http, "http://model-server/invocations"))


@pytest_asyncio.fixture
async def risk_service_returns_labels(label_transport) -> RiskService:
    async with httpx.AsyncClient(transport=label_transport) as http:
        yield RiskService(MLflowRiskClient(http, "http://model-server/invocations"))


@pytest_asyncio.fixture
async def client(risk_service) -> AsyncClient:
    """An async HTTP client that talks to the app in-process (no server needed)."""
    app.dependency_overrides[get_risk_service] = lambda: risk_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client_no_model_server(unreachable_transport) -> AsyncClient:
    async with httpx.AsyncClient(transport=unreachable_transport) as http:
        service = RiskService(MLflowRiskClient(http, "http://model-server/invocations"))
        app.dependency_overrides[get_risk_service] = lambda: service
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                yield ac
        finally:
            app.dependency_overrides.clear()
