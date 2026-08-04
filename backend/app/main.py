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
from .services.mlflow_client import MLflowRiskClient
from .services.risk import RiskService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the HTTP client for the process lifetime.

    One pooled client shared by every request, rather than one per request: the
    five model calls per question reuse warm connections instead of re-running a
    TCP handshake each time.
    """
    async with httpx.AsyncClient(timeout=settings.mlflow_timeout_s) as http:
        app.state.http_client = http
        app.state.risk_service = RiskService(
            MLflowRiskClient(http, settings.mlflow_url)
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(v1_router)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    return app


app = create_app()
