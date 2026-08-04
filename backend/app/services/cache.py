"""Risk cache.

The cache key is the **feature payload hash** — a digest of the model name, the
model version, and the exact feature vector — not the patient id. That choice is
what makes caching safe here rather than merely fast:

* The models are deterministic and pure, so identical inputs imply an identical
  probability. A hit cannot serve a *wrong* answer, only an older ``computed_at``.
* Re-registering a model with new coefficients changes ``model_version``, which
  changes the key, so stale results from a previous model cannot survive a
  deployment.
* It is the same hash the append-dedupe uses. One primitive, two requirements.

What a cache CAN get wrong is provenance: presenting a value computed an hour ago
as if it were computed now. So a hit is labelled — ``RiskResult.source`` is
``cache`` and ``computed_at`` is the original timestamp, not the request time.

Off by default (``CACHE_BACKEND=none``). At five features and a logistic
regression the compute saved is around a millisecond; the value here is the
pattern and the shared-key insight, not the latency.
"""

from __future__ import annotations

import json
from typing import Protocol

from ..core.config import settings


class RiskCache(Protocol):
    async def get(self, key: str) -> dict | None: ...
    async def set(self, key: str, value: dict) -> None: ...
    async def close(self) -> None: ...


class NullCache:
    """Default. Every lookup misses, nothing is stored."""

    async def get(self, key: str) -> dict | None:
        return None

    async def set(self, key: str, value: dict) -> None:
        return None

    async def close(self) -> None:
        return None


class RedisCache:
    """Cache-aside over Redis.

    Deliberately fail-open: if Redis is unreachable, a lookup misses and a store
    is dropped, so the request still completes by calling the model. A cache
    outage should degrade latency, never availability — and never correctness,
    since the fallback is to recompute.
    """

    def __init__(self, url: str | None = None, ttl_s: int | None = None) -> None:
        import redis.asyncio as redis  # imported lazily; unused when CACHE=none

        self._redis = redis.from_url(
            url or settings.redis_url, encoding="utf-8", decode_responses=True
        )
        self._ttl = ttl_s if ttl_s is not None else settings.cache_ttl_s

    async def get(self, key: str) -> dict | None:
        try:
            raw = await self._redis.get(key)
        except Exception:  # noqa: BLE001 - see fail-open note above
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    async def set(self, key: str, value: dict) -> None:
        try:
            await self._redis.set(key, json.dumps(value), ex=self._ttl)
        except Exception:  # noqa: BLE001
            return None

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except Exception:  # noqa: BLE001
            return None


def cache_key(model_name: str, inputs_hash: str) -> str:
    # inputs_hash already covers model name and version; the prefix is for
    # human legibility when inspecting Redis directly.
    return f"risk:{model_name}:{inputs_hash}"


def build_cache(backend: str | None = None) -> RiskCache:
    """Pick a cache from configuration. Defaults to no cache."""
    choice = backend or settings.cache_backend
    if choice == "redis":
        return RedisCache()
    return NullCache()
