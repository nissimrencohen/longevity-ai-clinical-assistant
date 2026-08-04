"""FastAPI dependencies.

The ``RiskService`` is built once during app startup (see ``app.main``) and shared,
because it holds a single ``httpx.AsyncClient`` — creating a client per request
throws away connection pooling and is the usual reason "async" code turns out slow.

Tests override ``get_risk_service`` to inject a service whose HTTP transport is
mocked, so the whole request path can be exercised without a live model server.
"""

from __future__ import annotations

from fastapi import Request

from ..services.risk import RiskService


def get_risk_service(request: Request) -> RiskService:
    return request.app.state.risk_service
