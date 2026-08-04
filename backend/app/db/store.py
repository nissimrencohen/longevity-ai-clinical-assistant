"""Storage abstraction over the clinical database.

One protocol, two implementations. The service layer talks only to
``ClinicalStore`` and does not know which backend is behind it, which is what
lets the same test suite and the same eval harness run against both.

* ``SqliteStore``   — the shipped fixture. Default, needs no services, and is
                      what the eval gold values are anchored to.
* ``PostgresStore`` — what the containerised stack runs. Row-level MVCC instead
                      of a database-wide write lock, and the dedupe rule becomes
                      an atomic ``INSERT ... ON CONFLICT DO NOTHING`` instead of
                      a read-then-write race.

The append semantics are identical either way, and
``backend/tests/test_risk_service.py`` runs its assertions against both.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import aiosqlite
from sqlalchemy import insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from ..core.config import settings
from .sqlite import open_db
from .models import biomarkers as biomarkers_table
from .models import demographics as demographics_table
from .models import risks as risks_table


@dataclass(frozen=True)
class RiskRow:
    """One row destined for the ``risks`` append log."""

    patient_id: str
    risk_code: str
    probability: float
    risk_band: str
    model_name: str
    model_version: str | None
    time_horizon_years: int | None
    computed_at: str
    inputs_json: str
    inputs_hash: str


class ClinicalStore(Protocol):
    """What the risk service needs from a database."""

    async def fetch_record(
        self, patient_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """``(demographics, biomarkers)`` as plain dicts, or ``None`` if unknown."""
        ...

    async def fetch_last_inputs_hashes(self, patient_id: str) -> dict[str, str | None]:
        """Latest stored inputs hash per risk_code."""
        ...

    async def append_risks(self, rows: Iterable[RiskRow]) -> list[RiskRow]:
        """Insert rows, skipping any whose (patient, model, inputs) already exist."""
        ...

    async def fetch_trends(
        self, patient_id: str, *, limit_per_risk: int = 12
    ) -> dict[str, list[dict[str, Any]]]:
        """The append log as an ascending series per risk code."""
        ...

    async def close(self) -> None:
        ...


def direction(points: list[Mapping[str, Any]]) -> str:
    """Describe a series as improving / worsening / stable.

    Compares the newest point against the previous one. The 0.01 dead-band stops
    trivial numerical wobble from being reported to a clinician as a change.
    """
    if len(points) < 2:
        return "insufficient_history"
    delta = float(points[-1]["probability"]) - float(points[-2]["probability"])
    if delta > 0.01:
        return "worsening"
    if delta < -0.01:
        return "improving"
    return "stable"


def _trim(
    trends: dict[str, list[dict[str, Any]]], limit_per_risk: int
) -> dict[str, list[dict[str, Any]]]:
    # Keep the most recent N points, preserving ascending order — the trend is
    # for reading a direction, not for plotting five years of history.
    return {code: points[-limit_per_risk:] for code, points in trends.items()}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


class SqliteStore:
    """The shipped fixture, over aiosqlite.

    Connections are opened per operation rather than held: the risk endpoint does
    network I/O to five models between its read and its write, and holding a
    write-capable SQLite connection across that turns a slow model server into a
    locked database.
    """

    def __init__(self, db_path: Any | None = None) -> None:
        self._db_path = db_path

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        # Resolve the path per call rather than at construction: tests monkeypatch
        # settings.patient_db_path to a temp copy after the service is built.
        async with open_db(self._db_path) as conn:
            yield conn

    async def fetch_record(
        self, patient_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        async with self._connect() as db:
            async with db.execute(
                "SELECT * FROM demographics WHERE patient_id = ?", (patient_id,)
            ) as cur:
                dem = await cur.fetchone()
            if dem is None:
                return None
            # biomarkers is modelled as a history table; always take the newest.
            async with db.execute(
                "SELECT * FROM biomarkers WHERE patient_id = ? "
                "ORDER BY measured_at DESC, id DESC LIMIT 1",
                (patient_id,),
            ) as cur:
                bio = await cur.fetchone()
        return dict(dem), (dict(bio) if bio is not None else {})

    async def fetch_last_inputs_hashes(self, patient_id: str) -> dict[str, str | None]:
        async with self._connect() as db:
            async with db.execute(
                """
                SELECT r.risk_code, r.inputs_json
                FROM risks r
                JOIN (
                    SELECT risk_code, MAX(id) AS max_id
                    FROM risks WHERE patient_id = ? GROUP BY risk_code
                ) latest ON latest.max_id = r.id
                """,
                (patient_id,),
            ) as cur:
                rows = await cur.fetchall()

        hashes: dict[str, str | None] = {}
        for row in rows:
            raw = row["inputs_json"]
            digest: str | None = None
            if raw:
                try:
                    digest = json.loads(raw).get("inputs_hash")
                except (ValueError, AttributeError):
                    # Seeded history carries NULL inputs_json; anything
                    # unparseable counts as "unknown inputs" so we re-append.
                    digest = None
            hashes[row["risk_code"]] = digest
        return hashes

    async def append_risks(self, rows: Iterable[RiskRow]) -> list[RiskRow]:
        written = list(rows)
        if not written:
            return []
        async with self._connect() as db:
            await db.executemany(
                """
                INSERT INTO risks (
                    patient_id, risk_code, probability, risk_band, model_name,
                    model_version, time_horizon_years, computed_at, inputs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.patient_id, r.risk_code, r.probability, r.risk_band,
                        r.model_name, r.model_version, r.time_horizon_years,
                        r.computed_at, r.inputs_json,
                    )
                    for r in written
                ],
            )
            await db.commit()
        return written

    async def fetch_trends(
        self, patient_id: str, *, limit_per_risk: int = 12
    ) -> dict[str, list[dict[str, Any]]]:
        async with self._connect() as db:
            async with db.execute(
                """
                SELECT risk_code, computed_at, probability, risk_band
                FROM risks WHERE patient_id = ?
                ORDER BY risk_code ASC, computed_at ASC, id ASC
                """,
                (patient_id,),
            ) as cur:
                rows = await cur.fetchall()

        trends: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            trends.setdefault(row["risk_code"], []).append(
                {
                    "computed_at": row["computed_at"],
                    "probability": float(row["probability"]),
                    "risk_band": row["risk_band"],
                }
            )
        return _trim(trends, limit_per_risk)

    async def close(self) -> None:  # nothing pooled
        return None


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


class PostgresStore:
    """SQLAlchemy 2.0 async over asyncpg."""

    def __init__(self, dsn: str | None = None, *, engine: AsyncEngine | None = None) -> None:
        self._engine = engine or create_async_engine(
            dsn or settings.postgres_dsn,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,  # a recycled connection to a restarted DB is dead
        )
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

    async def fetch_record(
        self, patient_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        async with self._session() as session:
            dem = (
                await session.execute(
                    select(demographics_table).where(
                        demographics_table.c.patient_id == patient_id
                    )
                )
            ).mappings().first()
            if dem is None:
                return None

            bio = (
                await session.execute(
                    select(biomarkers_table)
                    .where(biomarkers_table.c.patient_id == patient_id)
                    .order_by(
                        biomarkers_table.c.measured_at.desc(),
                        biomarkers_table.c.id.desc(),
                    )
                    .limit(1)
                )
            ).mappings().first()

        return dict(dem), (dict(bio) if bio is not None else {})

    async def fetch_last_inputs_hashes(self, patient_id: str) -> dict[str, str | None]:
        # DISTINCT ON is the natural Postgres spelling of "latest row per group".
        statement = text(
            """
            SELECT DISTINCT ON (risk_code) risk_code, inputs_hash
            FROM risks
            WHERE patient_id = :pid
            ORDER BY risk_code, id DESC
            """
        )
        async with self._session() as session:
            rows = (await session.execute(statement, {"pid": patient_id})).all()
        return {code: digest for code, digest in rows}

    async def append_risks(self, rows: Iterable[RiskRow]) -> list[RiskRow]:
        """Atomic dedupe: the unique index decides, not a prior SELECT.

        Returns only the rows that were actually inserted, so a concurrent
        duplicate request reports ``persisted=False`` truthfully rather than
        claiming a write that the database rejected.
        """
        pending = list(rows)
        if not pending:
            return []

        values = [
            {
                "patient_id": r.patient_id,
                "risk_code": r.risk_code,
                "probability": r.probability,
                "risk_band": r.risk_band,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "time_horizon_years": r.time_horizon_years,
                "computed_at": r.computed_at,
                "inputs_json": r.inputs_json,
                "inputs_hash": r.inputs_hash,
            }
            for r in pending
        ]

        statement = (
            pg_insert(risks_table)
            .values(values)
            .on_conflict_do_nothing(
                index_elements=["patient_id", "model_name", "inputs_hash"],
                # The unique index is PARTIAL (WHERE inputs_hash IS NOT NULL, so
                # the seeded history rows are exempt). Postgres will only infer a
                # partial index when the statement repeats its predicate —
                # without index_where it raises "there is no unique or exclusion
                # constraint matching the ON CONFLICT specification", even though
                # the index plainly exists.
                index_where=text("inputs_hash IS NOT NULL"),
            )
            .returning(risks_table.c.risk_code)
        )
        async with self._session() as session:
            inserted = {code for (code,) in (await session.execute(statement)).all()}
            await session.commit()

        return [r for r in pending if r.risk_code in inserted]

    async def fetch_trends(
        self, patient_id: str, *, limit_per_risk: int = 12
    ) -> dict[str, list[dict[str, Any]]]:
        async with self._session() as session:
            rows = (
                await session.execute(
                    select(
                        risks_table.c.risk_code,
                        risks_table.c.computed_at,
                        risks_table.c.probability,
                        risks_table.c.risk_band,
                    )
                    .where(risks_table.c.patient_id == patient_id)
                    .order_by(
                        risks_table.c.risk_code.asc(),
                        risks_table.c.computed_at.asc(),
                        risks_table.c.id.asc(),
                    )
                )
            ).all()

        trends: dict[str, list[dict[str, Any]]] = {}
        for code, computed_at, probability, band in rows:
            trends.setdefault(code, []).append(
                {
                    "computed_at": computed_at,
                    "probability": float(probability),
                    "risk_band": band,
                }
            )
        return _trim(trends, limit_per_risk)

    async def close(self) -> None:
        await self._engine.dispose()


def build_store(backend: str | None = None) -> ClinicalStore:
    """Pick a store from configuration. Defaults to SQLite."""
    choice = backend or settings.db_backend
    if choice == "postgres":
        return PostgresStore()
    return SqliteStore()


__all__ = [
    "ClinicalStore",
    "PostgresStore",
    "RiskRow",
    "SqliteStore",
    "build_store",
    "direction",
    "insert",  # re-exported for tests that build raw statements
]
