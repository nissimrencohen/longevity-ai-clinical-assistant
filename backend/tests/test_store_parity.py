"""Store-layer tests: cache semantics, and SQLite/Postgres behavioural parity.

The Postgres tests skip when no database is reachable, so `uv run pytest` on a
clean clone with no services still passes. Run them with:

    make up-debug          # publishes Postgres on 55432
    POSTGRES_DSN=postgresql+asyncpg://clinic:clinic@127.0.0.1:55432/clinic uv run pytest
"""

from __future__ import annotations

import json
import os

import pytest

from backend.app.db.store import RiskRow, SqliteStore, direction
from backend.app.services.cache import NullCache, cache_key

# ---------------------------------------------------------------------------
# Cache semantics
# ---------------------------------------------------------------------------


def test_cache_key_is_content_addressed_not_patient_addressed() -> None:
    """Two patients with identical inputs share a key; that is the point.

    The models are pure, so identical feature vectors imply identical outputs.
    Keying on patient_id instead would cache the same computation N times and,
    worse, would not invalidate when a patient's biomarkers changed.
    """
    assert cache_key("framingham_ckd", "abc123") == cache_key("framingham_ckd", "abc123")
    assert cache_key("framingham_ckd", "abc123") != cache_key("ada_t2dm", "abc123")


async def test_null_cache_always_misses() -> None:
    cache = NullCache()
    await cache.set("k", {"probability": 0.5})
    assert await cache.get("k") is None


async def test_cache_hit_reports_original_timestamp(risk_service_cached) -> None:
    """A hit must not present an old value as freshly computed.

    This is the failure mode that makes caching a clinical number risky: the
    number is right, but its provenance is a lie. `source` and `computed_at`
    together keep the response honest.
    """
    service, cache = risk_service_cached

    first = await service.get_current_risks("P001")
    assert all(r.source == "fresh" for r in first.risks)
    assert len(cache.store) == 5  # one entry per model

    second = await service.get_current_risks("P001")
    assert all(r.source == "cache" for r in second.risks)
    for a, b in zip(first.risks, second.risks, strict=True):
        assert a.probability == pytest.approx(b.probability)
        assert b.computed_at == a.computed_at  # original time, not now


def test_model_version_bump_invalidates_the_cache_key() -> None:
    """Re-registering a model must not be able to read the old version's answers.

    inputs_hash covers model name AND version, so identical features under a new
    version hash differently — which is what makes it safe to cache a clinical
    number indefinitely.
    """
    import dataclasses

    from backend.app.services.features import SPECS_BY_RISK_CODE
    from backend.app.services.risk import _inputs_hash

    spec = SPECS_BY_RISK_CODE["CKD"]
    payload = {feature: 1.0 for feature in spec.features}

    original = _inputs_hash(spec, payload)
    bumped = _inputs_hash(dataclasses.replace(spec, model_version="2.0.0"), payload)

    assert original != bumped
    assert cache_key(spec.model_name, original) != cache_key(spec.model_name, bumped)


def test_changed_features_change_the_cache_key() -> None:
    """The whole point of a content-addressed key: new labs, new entry."""
    from backend.app.services.features import SPECS_BY_RISK_CODE
    from backend.app.services.risk import _inputs_hash

    spec = SPECS_BY_RISK_CODE["CKD"]
    baseline = {feature: 1.0 for feature in spec.features}
    changed = {**baseline, "egfr": 52.0}

    assert _inputs_hash(spec, baseline) != _inputs_hash(spec, changed)


async def test_cache_failure_does_not_break_the_request(risk_service_broken_cache) -> None:
    """A cache outage must degrade latency, never availability or correctness."""
    response = await risk_service_broken_cache.get_current_risks("P001")
    assert len(response.risks) == 5
    assert all(r.source == "fresh" for r in response.risks)


# ---------------------------------------------------------------------------
# Trend direction (backend-independent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        ([], "insufficient_history"),
        ([{"probability": 0.4}], "insufficient_history"),
        ([{"probability": 0.39}, {"probability": 0.45}], "worsening"),
        ([{"probability": 0.45}, {"probability": 0.39}], "improving"),
        ([{"probability": 0.40}, {"probability": 0.405}], "stable"),
    ],
)
def test_direction(series: list[dict], expected: str) -> None:
    assert direction(series) == expected


def test_direction_dead_band_ignores_numerical_wobble() -> None:
    """A 0.005 move is not a clinical change and must not be reported as one."""
    assert direction([{"probability": 0.30}, {"probability": 0.305}]) == "stable"
    assert direction([{"probability": 0.30}, {"probability": 0.32}]) == "worsening"


# ---------------------------------------------------------------------------
# SQLite <-> Postgres parity
# ---------------------------------------------------------------------------

POSTGRES_DSN = os.getenv("POSTGRES_DSN")


async def _postgres_store():
    from backend.app.db.store import PostgresStore

    store = PostgresStore(POSTGRES_DSN)
    try:
        await store.fetch_record("P001")
    except Exception as exc:  # noqa: BLE001
        await store.close()
        pytest.skip(f"Postgres not reachable at {POSTGRES_DSN}: {exc}")
    return store


requires_postgres = pytest.mark.skipif(
    not POSTGRES_DSN,
    reason="set POSTGRES_DSN to run Postgres parity tests (see module docstring)",
)


@requires_postgres
async def test_same_patient_record_from_both_backends() -> None:
    """The two backends must return the same clinical picture, field for field."""
    pg = await _postgres_store()
    try:
        sqlite_record = await SqliteStore().fetch_record("P004")
        pg_record = await pg.fetch_record("P004")
    finally:
        await pg.close()

    assert sqlite_record is not None and pg_record is not None
    sqlite_dem, sqlite_bio = sqlite_record
    pg_dem, pg_bio = pg_record

    for field in ("patient_id", "first_name", "last_name", "date_of_birth", "sex",
                  "height_cm", "weight_kg", "hx_diabetes", "gestational_diabetes"):
        assert sqlite_dem[field] == pg_dem[field], field
    for field in ("egfr_ml_min_1_73m2", "systolic_bp", "urine_dipstick_protein"):
        assert sqlite_bio[field] == pg_bio[field], field


@requires_postgres
async def test_unknown_patient_is_none_in_postgres_too() -> None:
    pg = await _postgres_store()
    try:
        assert await pg.fetch_record("P999") is None
    finally:
        await pg.close()


@requires_postgres
async def test_postgres_dedupe_is_atomic() -> None:
    """The unique index — not a prior SELECT — decides whether a row is written.

    Appending the same (patient, model, inputs_hash) twice must insert once. On
    SQLite this is enforced by a read-then-write check with a race window; here
    it is a constraint, so concurrent duplicates cannot both land.
    """
    import uuid

    from sqlalchemy import text as sa_text

    pg = await _postgres_store()
    # A fresh hash per run. Postgres persists between test runs, so a fixed hash
    # meant the "first" insert had already happened on a previous run and the
    # test failed against a perfectly working constraint.
    digest = f"parity-{uuid.uuid4().hex}"
    row = RiskRow(
        patient_id="P001",
        risk_code="CKD",
        probability=0.123,
        risk_band="borderline",
        model_name="framingham_ckd",
        model_version="test",
        time_horizon_years=10,
        computed_at="2026-08-05T00:00:00+00:00",
        inputs_json=json.dumps({"inputs_hash": digest}),
        inputs_hash=digest,
    )
    try:
        first = await pg.append_risks([row])
        second = await pg.append_risks([row])
    finally:
        # Leave the database as we found it.
        async with pg._session() as session:  # noqa: SLF001 - test cleanup
            await session.execute(
                sa_text("DELETE FROM risks WHERE inputs_hash = :h"), {"h": digest}
            )
            await session.commit()
        await pg.close()

    assert len(first) == 1, "first append should insert"
    assert second == [], "identical inputs must not append a second row"
