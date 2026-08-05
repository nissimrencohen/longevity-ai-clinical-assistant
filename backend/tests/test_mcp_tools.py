"""MCP server startup guards.

Tool behaviour is covered by the eval harness (which calls the real server over
the wire) and by `test_security.py`. What is checked here is the one thing no
eval can reach: whether the process refuses to start in a configuration that
would be unsafe.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _server_module():
    sys.path.insert(0, str(REPO_ROOT / "mcp-server"))
    return importlib.import_module("server")


# ---------------------------------------------------------------------------
# The committed dev token must not survive into a real deployment
# ---------------------------------------------------------------------------


def test_dev_token_is_refused_outside_development(monkeypatch) -> None:
    """`dev-longevity-token-change-me` is public — it is in .env.example.

    That is deliberate, so a fresh clone runs. But in this server the token is
    also the IDENTITY: it carries the physician role, so shipping it to a real
    deployment would mean a publicly-known credential to a clinical API. A log
    warning gets read after the fact, if at all; refusing to boot cannot be
    missed.
    """
    server = _server_module()

    monkeypatch.setattr(server, "MCP_BEARER_TOKEN", server.DEV_BEARER_TOKEN)
    monkeypatch.setenv("APP_ENV", "production")

    with pytest.raises(SystemExit) as excinfo:
        server.main()
    assert "Refusing to start" in str(excinfo.value)


@pytest.mark.parametrize("app_env", ["dev", "development", "test", ""])
def test_local_development_still_works_with_the_default(monkeypatch, app_env) -> None:
    """The guard must not make a fresh clone harder to run — that is the point."""
    server = _server_module()

    monkeypatch.setattr(server, "MCP_BEARER_TOKEN", server.DEV_BEARER_TOKEN)
    monkeypatch.setenv("APP_ENV", app_env)
    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: None)

    server.main()  # must not raise


def test_a_real_token_starts_in_production(monkeypatch) -> None:
    """The guard keys on the DEFAULT value, not on APP_ENV alone."""
    server = _server_module()

    monkeypatch.setattr(server, "MCP_BEARER_TOKEN", "a-real-rotated-secret")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: None)

    server.main()  # must not raise
