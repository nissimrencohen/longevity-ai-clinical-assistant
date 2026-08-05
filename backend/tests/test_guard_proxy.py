"""Tests for the guard proxy itself, with a mocked OpenRouter upstream.

The policy is tested separately; this covers the wiring, which is where a guard
quietly stops guarding:

* streaming must be enforced too — that is the path LibreChat actually uses, and
  a guard that only covers non-streaming responses protects nothing in practice;
* tool-call turns carry no prose and must pass through untouched, or the agent
  loop breaks;
* a clean answer must come back byte-identical, so the guard is invisible when it
  has no opinion.

No network: the upstream is an httpx MockTransport.
"""

from __future__ import annotations

import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from guard import app as guard_app

PRESCRIBING = "His CVD risk is 0.44. I recommend starting atorvastatin 40 mg daily."
CLEAN = "His CVD risk is 0.44, in the high band, driven by age and smoking."


def _completion(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _sse_stream(content: str) -> str:
    """Upstream SSE for a message delivered in several deltas."""
    lines = []
    for piece in [content[i : i + 12] for i in range(0, len(content), 12)]:
        lines.append(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-test",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"content": piece}}],
                }
            )
        )
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def _mock_upstream(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://upstream.test/v1"
    )


@pytest.fixture
def guard_client():
    """Build the app with a stubbed upstream, bypassing lifespan."""

    async def _make(upstream: httpx.AsyncClient) -> AsyncClient:
        guard_app.app.state.upstream = upstream
        return AsyncClient(
            transport=ASGITransport(app=guard_app.app), base_url="http://guard"
        )

    return _make


async def test_non_streaming_prescription_is_stripped(guard_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(PRESCRIBING))

    async with _mock_upstream(handler) as upstream:
        client = await guard_client(upstream)
        async with client:
            response = await client.post("/v1/chat/completions", json={"model": "m"})

    body = response.json()
    content = body["choices"][0]["message"]["content"]

    assert response.status_code == 200
    assert "atorvastatin" not in content.lower()
    assert "0.44" in content, "the useful clinical summary must survive"
    assert "safety guard" in content
    assert body["x_clinical_guard"][0]["triggered"] is True


async def test_non_streaming_clean_answer_is_untouched(guard_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(CLEAN))

    async with _mock_upstream(handler) as upstream:
        client = await guard_client(upstream)
        async with client:
            response = await client.post("/v1/chat/completions", json={"model": "m"})

    body = response.json()
    assert body["choices"][0]["message"]["content"] == CLEAN
    assert "x_clinical_guard" not in body


async def test_streaming_prescription_is_stripped(guard_client) -> None:
    """The path LibreChat actually uses.

    Tokens cannot be retracted once streamed, so the proxy buffers and only then
    emits. A guard that covered non-streaming alone would protect nothing real.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_sse_stream(PRESCRIBING),
            headers={"content-type": "text/event-stream"},
        )

    async with _mock_upstream(handler) as upstream:
        client = await guard_client(upstream)
        async with client:
            response = await client.post(
                "/v1/chat/completions", json={"model": "m", "stream": True}
            )
            text = response.text

    delivered = "".join(
        json.loads(line[5:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
        for line in text.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    )

    assert "atorvastatin" not in delivered.lower()
    assert "0.44" in delivered
    assert text.rstrip().endswith("[DONE]")


async def test_streaming_clean_answer_is_replayed_verbatim(guard_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_sse_stream(CLEAN),
            headers={"content-type": "text/event-stream"},
        )

    async with _mock_upstream(handler) as upstream:
        client = await guard_client(upstream)
        async with client:
            response = await client.post(
                "/v1/chat/completions", json={"model": "m", "stream": True}
            )
            text = response.text

    delivered = "".join(
        json.loads(line[5:]).get("choices", [{}])[0].get("delta", {}).get("content", "")
        for line in text.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    )
    assert delivered == CLEAN


async def test_tool_call_turn_passes_through(guard_client) -> None:
    """No prose to guard — and mangling this would break the agent loop."""
    tool_turn = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_current_risks",
                                "arguments": '{"patient_id":"P002"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=tool_turn)

    async with _mock_upstream(handler) as upstream:
        client = await guard_client(upstream)
        async with client:
            response = await client.post("/v1/chat/completions", json={"model": "m"})

    body = response.json()
    calls = body["choices"][0]["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "get_current_risks"
    assert "x_clinical_guard" not in body


async def test_upstream_error_is_surfaced_not_swallowed(guard_client) -> None:
    """A guard that hides provider failures would be worse than none."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "insufficient credits"}})

    async with _mock_upstream(handler) as upstream:
        client = await guard_client(upstream)
        async with client:
            response = await client.post("/v1/chat/completions", json={"model": "m"})

    assert response.status_code == 402
    assert "insufficient credits" in json.dumps(response.json())
