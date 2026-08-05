"""FastAPI dependencies.

The ``RiskService`` is built once during app startup (see ``app.main``) and shared,
because it holds a single ``httpx.AsyncClient``, one database engine and one cache
connection — creating those per request throws away pooling and is the usual
reason "async" code turns out slow.

The Actor is per request. It comes from headers the MCP server sets from its
VERIFIED bearer token, never from anything the model or the user can type. The
MCP server is the trust boundary; the backend trusts it because they share a
private network and nothing else can reach the backend (it publishes no host
port). That assumption is written down here because it is exactly the kind of
thing that silently stops being true.
"""

from __future__ import annotations

from fastapi import Request

from ..core.security import DEFAULT_ACTOR, Actor, Role
from ..services.risk import RiskService


def get_risk_service(request: Request) -> RiskService:
    return request.app.state.risk_service


def get_actor(request: Request) -> Actor:
    """Identify the caller from MCP-set headers, falling back to the default.

    Absent headers mean a direct call (host mode, a curl, the test suite) and
    resolve to the default clinic physician — which is the brief's behaviour and
    keeps every existing path working unchanged.
    """
    actor_id = request.headers.get("X-Actor-Id")
    raw_role = request.headers.get("X-Actor-Role")

    if not actor_id and not raw_role:
        return DEFAULT_ACTOR

    try:
        role = Role(raw_role) if raw_role else DEFAULT_ACTOR.role
    except ValueError:
        # An unrecognised role is a misconfiguration, not an escalation route:
        # fall back to the least surprising identity rather than granting more.
        role = DEFAULT_ACTOR.role

    return Actor(
        actor_id=actor_id or DEFAULT_ACTOR.actor_id,
        role=role,
        clinic_id=request.headers.get("X-Clinic-Id", DEFAULT_ACTOR.clinic_id),
    )
