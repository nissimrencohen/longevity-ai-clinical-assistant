"""Minimal OpenAI-compatible client for OpenRouter, built on httpx.

Deliberately not the `openai` SDK: httpx is already a dependency, the surface we
need is one endpoint, and keeping the tool-calling loop explicit means the whole
agent path is readable in one file rather than hidden behind a client library.

The API key is read from LibreChat's .env if it is not in this repo's — the
assignment keeps the key there, and duplicating a secret into a second file just
to run the evals would be a bad habit to encode.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx

MAX_RETRIES = 6
RETRY_BACKOFF_S = 3.0
# Free-tier models are rate-limited per minute. Pacing requests costs wall-clock
# time but converts 429 storms into completed runs, which is the trade worth
# making: an errored run yields no signal at all.
REQUEST_PACING_S = float(os.getenv("EVAL_PACING_S", "1.5"))
# 429 rate limit, 408/409 transient, 5xx provider hiccups. NOT 402 (no credit) or
# 401 (bad key) — retrying those just wastes time on a condition that will not fix
# itself.
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Defaults are FREE, tool-capable models so `--tier b` runs on an account with no
# credit — the harness should not require a funded account to be reproducible.
# Pass --model for a stronger one (e.g. anthropic/claude-haiku-4.5); the reported
# pass rate is a property of the model as much as of the system, so the model id
# is always recorded in the results file.
DEFAULT_MODEL = os.getenv("EVAL_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
DEFAULT_JUDGE_MODEL = os.getenv(
    "EVAL_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)

# Answers here are short (a risk panel and a sentence of context). Capping keeps
# the credit reservation small - see the note in chat().
DEFAULT_MAX_TOKENS = int(os.getenv("EVAL_MAX_TOKENS", "1200"))

# Searched in order for OPENROUTER_KEY.
_ENV_CANDIDATES = (
    REPO_ROOT / ".env",
    REPO_ROOT / "librechat" / ".env",
    REPO_ROOT.parent / "LibreChat" / ".env",
)


class OpenRouterError(RuntimeError):
    pass


def find_api_key() -> str | None:
    key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
    if key:
        return key.strip()

    for path in _ENV_CANDIDATES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_KEY=") or line.startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    return None


class OpenRouterClient:
    """One method: send messages (+ optional tools), get the assistant message back."""

    def __init__(self, api_key: str, *, timeout: float = 120.0) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OpenRouterClient":
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                # OpenRouter asks callers to identify themselves.
                "HTTP-Referer": "https://github.com/nissimrencohen/longevity-ai-clinical-assistant",
                "X-Title": "Longevity Clinical AI - eval harness",
            },
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, Any]:
        if self._client is None:
            raise OpenRouterError("client used outside its async context")

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # Always send an explicit cap. Some models default to a very large
            # max_tokens, and OpenRouter reserves the full amount against the
            # account balance up front - which fails with HTTP 402 on a small
            # balance even though the actual reply is a few hundred tokens.
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # Free-tier models rate-limit aggressively; a 429 mid-suite would
        # otherwise be recorded as a model failure. Retry with backoff, and let
        # anything still failing surface as a run error (never as a wrong answer).
        last_error = ""
        for attempt in range(MAX_RETRIES):
            if REQUEST_PACING_S:
                await asyncio.sleep(REQUEST_PACING_S)

            response = await self._client.post("/chat/completions", json=body)

            if response.status_code == httpx.codes.OK:
                payload = response.json()
                if "choices" in payload:
                    return payload["choices"][0]["message"]

                # OpenRouter also reports upstream failures INSIDE a 200 body
                # (e.g. the provider rate-limited us). Treat those like the
                # equivalent HTTP status rather than as a malformed reply.
                embedded = payload.get("error") or {}
                code = embedded.get("code")
                last_error = f"upstream error {code}: {str(embedded.get('message'))[:300]}"
                if code not in _RETRYABLE or attempt == MAX_RETRIES - 1:
                    break
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in _RETRYABLE or attempt == MAX_RETRIES - 1:
                    break

            await asyncio.sleep(RETRY_BACKOFF_S * (2**attempt))

        raise OpenRouterError(f"{last_error} (after {MAX_RETRIES} attempts)")
