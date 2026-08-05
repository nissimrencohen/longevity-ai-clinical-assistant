"""Clinical safety guard — an OpenAI-compatible proxy in front of OpenRouter.

WHY A PROXY. The obvious place for an output guard is the backend or the MCP
server, and both are wrong: neither ever sees the assistant's prose. MCP carries
tool calls; the model's answer goes straight from OpenRouter to LibreChat and
then to the clinician. Putting a "guard" in the backend would be theatre.

LibreChat's custom endpoints take a configurable `baseURL`, which gives a real
interception point. Point it here instead of at OpenRouter and every token the
clinician sees has passed through policy:

    LibreChat ──▶ guard :8080 ──▶ OpenRouter
                    │
                    └─ enforce(), audit

The eval harness points here too (OPENROUTER_BASE_URL), so Tier B measures the
guarded path rather than an unguarded one.

STREAMING IS BUFFERED, DELIBERATELY. You cannot retract tokens already streamed
to a browser, so a guard that inspects mid-stream cannot enforce anything. The
proxy accumulates the upstream stream, applies policy to the complete message,
then emits. That costs token-by-token responsiveness and it is the right trade for a
clinical control: a partial prescription flashing on screen before being
retracted is worse than a slightly later answer.

Run:
    uv run uvicorn guard.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .policy import enforce

logger = logging.getLogger("guard")

UPSTREAM_BASE_URL = os.getenv("GUARD_UPSTREAM_URL", "https://openrouter.ai/api/v1")
UPSTREAM_TIMEOUT_S = float(os.getenv("GUARD_TIMEOUT_S", "180"))
FALLBACK_KEY = os.getenv("OPENROUTER_KEY", "")

# Headers we must not forward verbatim (hop-by-hop or recomputed downstream).
_STRIP_REQUEST_HEADERS = {"host", "content-length", "accept-encoding", "connection"}
_STRIP_RESPONSE_HEADERS = {
    "content-length", "content-encoding", "transfer-encoding", "connection",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient(
        base_url=UPSTREAM_BASE_URL, timeout=UPSTREAM_TIMEOUT_S
    ) as client:
        app.state.upstream = client
        yield


app = FastAPI(title="Clinical Safety Guard", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "clinical-safety-guard", "upstream": UPSTREAM_BASE_URL}


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _STRIP_REQUEST_HEADERS
    }
    if "authorization" not in {k.lower() for k in headers} and FALLBACK_KEY:
        headers["Authorization"] = f"Bearer {FALLBACK_KEY}"
    return headers


def _audit(verdict, *, streaming: bool) -> None:
    """Record every intervention. A guard nobody can review is not a control."""
    if not verdict.triggered:
        return
    logger.warning(
        "guard.blocked %s",
        json.dumps({"streaming": streaming, **verdict.to_audit()}, ensure_ascii=False),
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except ValueError:
        payload = {}

    upstream: httpx.AsyncClient = request.app.state.upstream
    headers = _forward_headers(request)
    streaming = bool(payload.get("stream"))

    if not streaming:
        response = await upstream.post("/chat/completions", content=body, headers=headers)
        if response.status_code != httpx.codes.OK:
            return JSONResponse(
                status_code=response.status_code,
                content=_safe_json(response),
            )
        return JSONResponse(content=_guard_response(response.json()))

    return StreamingResponse(
        _guarded_stream(upstream, body, headers),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text[:1000]}}


def _guard_response(data: dict) -> dict:
    """Apply policy to every choice in a non-streaming completion."""
    for choice in data.get("choices", []):
        message = choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue  # tool-call turns carry no prose to guard
        verdict = enforce(content)
        _audit(verdict, streaming=False)
        if verdict.triggered:
            message["content"] = verdict.text
            # Surface the intervention to any client that cares to look.
            data.setdefault("x_clinical_guard", []).append(verdict.to_audit())
    return data


async def _guarded_stream(
    upstream: httpx.AsyncClient, body: bytes, headers: dict[str, str]
) -> AsyncIterator[bytes]:
    """Buffer the upstream stream, apply policy, then emit.

    See the module docstring: mid-stream inspection cannot enforce anything,
    because the tokens are already on the clinician's screen.
    """
    raw_lines: list[str] = []
    content_parts: list[str] = []
    template: dict | None = None

    async with upstream.stream(
        "POST", "/chat/completions", content=body, headers=headers
    ) as response:
        if response.status_code != httpx.codes.OK:
            detail = (await response.aread()).decode("utf-8", "replace")[:1000]
            yield _sse({"error": {"message": detail, "code": response.status_code}})
            yield b"data: [DONE]\n\n"
            return

        async for line in response.aiter_lines():
            if not line:
                continue
            raw_lines.append(line)
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            template = template or chunk
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)

    verdict = enforce("".join(content_parts))
    _audit(verdict, streaming=True)

    if not verdict.triggered:
        # Nothing to change — replay upstream verbatim so tool calls, usage and
        # finish reasons survive untouched.
        for line in raw_lines:
            yield f"{line}\n\n".encode()
        if not any(line.strip() == "data: [DONE]" for line in raw_lines):
            yield b"data: [DONE]\n\n"
        return

    # Re-emit a clean single-chunk stream carrying the sanitised message.
    base = {
        "id": (template or {}).get("id", "guard"),
        "object": "chat.completion.chunk",
        "created": (template or {}).get("created", 0),
        "model": (template or {}).get("model", "guarded"),
    }
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    yield _sse({**base, "choices": [{"index": 0, "delta": {"content": verdict.text}}]})
    yield _sse(
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    yield b"data: [DONE]\n\n"


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


# Anything else on the API surface (model listing, generation metadata) is
# forwarded untouched — the guard only has an opinion about assistant prose.
@app.api_route(
    "/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
@app.api_route("/{path:path}", methods=["GET", "PUT", "DELETE", "PATCH"])
async def passthrough(request: Request, path: str) -> Any:
    upstream: httpx.AsyncClient = request.app.state.upstream
    body = await request.body()
    response = await upstream.request(
        request.method,
        f"/{path}",
        content=body or None,
        headers=_forward_headers(request),
        params=dict(request.query_params),
    )
    return JSONResponse(
        status_code=response.status_code,
        content=_safe_json(response),
        headers={
            k: v
            for k, v in response.headers.items()
            if k.lower() not in _STRIP_RESPONSE_HEADERS
        },
    )
