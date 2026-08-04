"""FastAPI application entrypoint.

Run (from the repo root):
    uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8001 --reload

``/health`` works out of the box. The two clinical endpoints under ``/api/v1``
need the MLflow router serving on :5001 (see GUIDE.md §4).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from .api.v1.endpoints import router as v1_router
from .core.config import settings
from .db.store import build_store
from .services.cache import build_cache
from .services.mlflow_client import MLflowRiskClient
from .services.risk import RiskService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the pooled resources for the process lifetime.

    One HTTP client, one database engine and one cache connection shared by every
    request. Creating them per request throws away pooling — the usual reason
    "async" code turns out slow — and for Postgres would mean a TCP handshake and
    an auth round trip on every question.
    """
    store = build_store()
    cache = build_cache()
    try:
        async with httpx.AsyncClient(timeout=settings.mlflow_timeout_s) as http:
            app.state.http_client = http
            app.state.store = store
            app.state.cache = cache
            app.state.risk_service = RiskService(
                MLflowRiskClient(http, settings.mlflow_url),
                store=store,
                cache=cache,
            )
            yield
    finally:
        await cache.close()
        await store.close()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(v1_router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        # Reports which data layer is live, so a misconfigured deployment is
        # visible from the healthcheck rather than from surprising results.
        return {
            "status": "ok",
            "service": settings.app_name,
            "db_backend": settings.db_backend,
            "cache_backend": settings.cache_backend,
        }

    return app


app = create_app()
