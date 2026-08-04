"""MCP client plumbing shared by both tiers.

Both tiers talk to the SAME MCP server the doctor's chat UI talks to — that is
the point. Tier A calls the tools directly; Tier B hands their schemas to an LLM
and lets it decide. Neither reaches past MCP into the backend, so a break in the
tool contract shows up in the evals rather than only in the UI.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.auth import BearerAuth

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

MCP_HOST = os.getenv("MCP_EVAL_HOST", "127.0.0.1")
MCP_PORT = os.getenv("MCP_PORT", "9100")
# No trailing slash: FastMCP 3.x serves /mcp and 307-redirects /mcp/ to it.
MCP_URL = os.getenv("MCP_EVAL_URL", f"http://{MCP_HOST}:{MCP_PORT}/mcp")
MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "dev-longevity-token-change-me")


class ToolCallError(RuntimeError):
    """The tool raised — e.g. unknown patient. Carries the model-visible message."""

    def __init__(self, tool: str, message: str) -> None:
        self.tool = tool
        self.message = message
        super().__init__(f"{tool}: {message}")


@asynccontextmanager
async def open_client():
    """Yield a connected, authenticated MCP client."""
    async with Client(MCP_URL, auth=BearerAuth(MCP_BEARER_TOKEN)) as client:
        yield client


async def call_tool(client: Client, name: str, arguments: dict[str, Any]) -> Any:
    """Call a tool, normalising failures into ToolCallError.

    A tool error is a legitimate, expected outcome here (unknown patient), not an
    infrastructure failure — the eval needs to see the message the model would
    see, so it can check the assistant relayed it instead of inventing data.
    """
    try:
        result = await client.call_tool(name, arguments)
    except Exception as exc:  # noqa: BLE001 - fastmcp raises several types
        raise ToolCallError(name, str(exc)) from exc

    if getattr(result, "is_error", False):
        raise ToolCallError(name, _result_text(result))
    return result.data


def _result_text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    parts = [getattr(block, "text", "") for block in content]
    return " ".join(p for p in parts if p) or str(result)


async def list_tool_schemas(client: Client) -> list[dict[str, Any]]:
    """MCP tool definitions in OpenAI `tools=[...]` shape, for Tier B."""
    tools = await client.list_tools()
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return schemas


async def probe() -> tuple[bool, str]:
    """Check the MCP server is reachable and authorised. Returns (ok, detail)."""
    try:
        async with open_client() as client:
            names = [t.name for t in await client.list_tools()]
        return True, f"{MCP_URL} ({len(names)} tools: {', '.join(names)})"
    except Exception as exc:  # noqa: BLE001
        return False, f"{MCP_URL} unreachable: {exc}"


def serialise(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
