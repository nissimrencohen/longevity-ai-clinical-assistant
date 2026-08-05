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

import glob
import importlib.util
import json
import pickle
import shutil
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
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


@lru_cache(maxsize=1)
def reference_vectors() -> dict[str, dict[str, float]]:
    """The same reference population the registered router serves.

    Read from the model artefact when it exists, falling back to the generator
    spec it was built from, so the mock cannot drift from the real thing.
    """
    matches = glob.glob(
        str(MODELS_DIR / "mlflow_risk_router" / "**" / "reference_vectors.json"),
        recursive=True,
    )
    if matches:
        return json.loads(Path(matches[0]).read_text(encoding="utf-8"))["vectors"]

    spec = importlib.util.spec_from_file_location(
        "generate_models", MODELS_DIR / "generate_models.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return {entry["metadata"]["model_name"]: entry["healthy"] for entry in module.MODELS}


def _router_handler(request: httpx.Request) -> httpx.Response:
    """Stand-in for the MLflow RiskRouter — same contract, same maths.

    Implements `explain` too. A mock that silently ignored it would let the
    explanation path go untested here while working in production (or worse, the
    reverse); test_mlflow_integration.py asserts the mocked and live paths agree.
    """
    body = json.loads(request.content)
    params = body.get("params", {})
    model_name = params["model"]
    split = body["dataframe_split"]

    model = load_model(model_name)
    frame = pd.DataFrame(split["data"], columns=split["columns"])
    missing = [c for c in model.feature_names_in_ if c not in frame.columns]
    if missing:
        return httpx.Response(400, json={"error": f"missing features: {missing}"})

    features = list(model.feature_names_in_)
    X = frame[features].astype("float64")
    probabilities = model.predict_proba(X)[:, 1]

    if not params.get("explain"):
        return httpx.Response(200, json={"predictions": probabilities.tolist()})

    coef = np.asarray(model.coef_[0], dtype="float64")
    intercept = float(model.intercept_[0])
    reference = reference_vectors()[model_name]
    ref_vec = np.array([float(reference[f]) for f in features], dtype="float64")
    base_value = intercept + float(coef @ ref_vec)

    predictions = []
    for position, (_, row) in enumerate(X.iterrows()):
        x = row.to_numpy(dtype="float64")
        phi = coef * (x - ref_vec)
        predictions.append(
            {
                "probability": float(probabilities[position]),
                "base_value": base_value,
                "reference_id": "healthy-anchor-v1",
                "model_name": model_name,
                "contributions": {f: float(v) for f, v in zip(features, phi)},
                "reference_values": {f: float(v) for f, v in zip(features, ref_vec)},
                "feature_values": {f: float(v) for f, v in zip(features, x)},
            }
        )
    return httpx.Response(200, json={"predictions": predictions})


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


class FakeCache:
    """In-memory stand-in for Redis, so cache SEMANTICS are tested without a server.

    What matters here is the contract (content-addressed keys, provenance on a
    hit), not Redis itself — and a test that needs a running service is a test
    that gets skipped.
    """

    def __init__(self) -> None:
        self.store: dict[str, dict] = {}

    async def get(self, key: str) -> dict | None:
        return self.store.get(key)

    async def set(self, key: str, value: dict) -> None:
        self.store[key] = value

    async def close(self) -> None:
        return None


class BrokenCache:
    """Every operation fails — mirrors Redis being down.

    RedisCache swallows its own errors, so this asserts the SERVICE stays correct
    when the cache is useless: degrade latency, never availability.
    """

    async def get(self, key: str) -> dict | None:
        raise RuntimeError("redis unreachable")

    async def set(self, key: str, value: dict) -> None:
        raise RuntimeError("redis unreachable")

    async def close(self) -> None:
        return None


class _SafeBrokenCache(BrokenCache):
    """BrokenCache behind the same fail-open guard RedisCache uses."""

    async def get(self, key: str) -> dict | None:
        try:
            return await super().get(key)
        except Exception:  # noqa: BLE001
            return None

    async def set(self, key: str, value: dict) -> None:
        try:
            await super().set(key, value)
        except Exception:  # noqa: BLE001
            return None


@pytest_asyncio.fixture
async def risk_service(mlflow_transport) -> RiskService:
    async with httpx.AsyncClient(transport=mlflow_transport) as http:
        yield RiskService(MLflowRiskClient(http, "http://model-server/invocations"))


@pytest_asyncio.fixture
async def risk_service_cached(mlflow_transport) -> tuple[RiskService, FakeCache]:
    cache = FakeCache()
    async with httpx.AsyncClient(transport=mlflow_transport) as http:
        yield (
            RiskService(
                MLflowRiskClient(http, "http://model-server/invocations"), cache=cache
            ),
            cache,
        )


@pytest_asyncio.fixture
async def risk_service_broken_cache(mlflow_transport) -> RiskService:
    async with httpx.AsyncClient(transport=mlflow_transport) as http:
        yield RiskService(
            MLflowRiskClient(http, "http://model-server/invocations"),
            cache=_SafeBrokenCache(),
        )


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
