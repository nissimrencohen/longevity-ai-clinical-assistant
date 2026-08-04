"""Async data access over the patient database (aiosqlite throughout).

Three collaborators, kept separate because they have genuinely different jobs:

* ``PatientRepository`` — reads demographics + the latest biomarker snapshot.
* ``RiskWriter``        — appends computed risks, with the dedupe rule.
* ``TrendBuilder``      — reads the append log back as a per-risk time series.

Nothing here imports FastAPI or Pydantic; these speak rows and dicts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import aiosqlite

RISK_INSERT = """
INSERT INTO risks (
    patient_id, risk_code, probability, risk_band, model_name, model_version,
    time_horizon_years, computed_at, inputs_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


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


class PatientRepository:
    """Reads the patient's current clinical picture."""

    async def fetch_demographics(
        self, db: aiosqlite.Connection, patient_id: str
    ) -> aiosqlite.Row | None:
        async with db.execute(
            "SELECT * FROM demographics WHERE patient_id = ?", (patient_id,)
        ) as cur:
            return await cur.fetchone()

    async def fetch_latest_biomarkers(
        self, db: aiosqlite.Connection, patient_id: str
    ) -> aiosqlite.Row | None:
        # biomarkers is modelled as a history table (one row per patient today, but
        # the schema anticipates longitudinal labs), so always take the newest.
        async with db.execute(
            """
            SELECT * FROM biomarkers
            WHERE patient_id = ?
            ORDER BY measured_at DESC, id DESC
            LIMIT 1
            """,
            (patient_id,),
        ) as cur:
            return await cur.fetchone()

    async def fetch_record(
        self, db: aiosqlite.Connection, patient_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Return ``(demographics, biomarkers)`` as plain dicts, or ``None``.

        Kept as two queries rather than one join: ``SELECT d.*, b.*`` collides on
        ``patient_id``/``id`` and quietly shadows columns, and this way a patient
        with no labs yet still resolves (404 is about the PATIENT existing, not
        about whether they have been drawn).
        """
        demographics = await self.fetch_demographics(db, patient_id)
        if demographics is None:
            return None
        biomarkers = await self.fetch_latest_biomarkers(db, patient_id)
        return dict(demographics), (dict(biomarkers) if biomarkers is not None else {})


class RiskWriter:
    """Appends computed risks — idempotently.

    The dedupe rule is the one the brief asks for: append only when the inputs
    changed since the last stored row for that (patient, model). Since the model
    is deterministic, identical inputs imply an identical probability, so skipping
    loses no information — it just keeps a doctor refreshing the page from
    spamming the trend with a flat line.

    That also makes the GET idempotent: repeated calls produce no observable state
    change, which is the part of "a GET that writes" that actually smells.

    The stored ``inputs_json`` carries the hash alongside the features, so the
    rule needs no schema change to the provided ``risks`` table.
    """

    async def fetch_last_inputs_hashes(
        self, db: aiosqlite.Connection, patient_id: str
    ) -> dict[str, str | None]:
        """Latest stored inputs hash per risk_code (``None`` when unknown)."""
        async with db.execute(
            """
            SELECT r.risk_code, r.inputs_json
            FROM risks r
            JOIN (
                SELECT risk_code, MAX(id) AS max_id
                FROM risks
                WHERE patient_id = ?
                GROUP BY risk_code
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
                    # Seeded history rows carry NULL inputs_json; anything
                    # unparseable is treated as "unknown inputs" so we re-append.
                    digest = None
            hashes[row["risk_code"]] = digest
        return hashes

    async def append(
        self, db: aiosqlite.Connection, rows: Iterable[RiskRow]
    ) -> list[RiskRow]:
        """Insert the given rows and commit. Returns what was written."""
        written = list(rows)
        if not written:
            return []
        await db.executemany(
            RISK_INSERT,
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


class TrendBuilder:
    """Reads the append log back as an ascending series per risk code."""

    async def fetch(
        self, db: aiosqlite.Connection, patient_id: str, *, limit_per_risk: int = 12
    ) -> dict[str, list[dict[str, Any]]]:
        async with db.execute(
            """
            SELECT risk_code, computed_at, probability, risk_band
            FROM risks
            WHERE patient_id = ?
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
        # Keep the most recent N points, preserving ascending order — the trend is
        # for reading a direction, not for plotting five years of history.
        return {code: points[-limit_per_risk:] for code, points in trends.items()}


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
