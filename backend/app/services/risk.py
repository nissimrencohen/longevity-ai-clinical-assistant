"""Risk computation service — the core of the backend.

``get_current_risks`` end to end:

  1. Load demographics + the latest biomarkers (404 if the patient is unknown).
  2. Derive the union of model inputs, then select each model's own payload.
  3. Call the five models CONCURRENTLY on the MLflow router (``asyncio.gather``).
  4. Band each probability.
  5. Append to the ``risks`` log — but only when the inputs actually changed,
     which keeps the trend meaningful and makes this GET idempotent.
  6. Read the log back as a trend, with an explicit direction per risk.

Design notes worth knowing:

* The SQLite connection is NOT held across the model calls. We read, close, do the
  network work, then reopen to write. Holding a write-capable connection open
  across network I/O is how you turn a 200 ms outage into a locked database.
* A model-server failure fails the whole request (502) rather than returning four
  of five risks. A partially-populated risk panel is worse than an honest error:
  the missing one reads as "not elevated".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import date, datetime, timezone
from typing import Any

import aiosqlite

from ..core.errors import IncompletePatientDataError, PatientNotFoundError
from ..db.repository import PatientRepository, RiskRow, RiskWriter, TrendBuilder, direction
from ..db.sqlite import open_db
from ..schemas import (
    BiomarkerSnapshot,
    BiomarkersResponse,
    RiskResult,
    RisksResponse,
    RiskTrendPoint,
)
from .banding import band
from .features import (
    CLINIC_TODAY,
    MODEL_SPECS,
    ModelSpec,
    build_payload,
    defaults_applied,
    derive_features,
    whole_years,
)
from .mlflow_client import MLflowRiskClient

DbFactory = Callable[[], AbstractAsyncContextManager[aiosqlite.Connection]]


def _inputs_hash(spec: ModelSpec, payload: dict[str, float]) -> str:
    """Stable fingerprint of exactly what was scored.

    Includes the model name and version, so re-registering a model with new
    coefficients correctly invalidates the dedupe (and, from Phase 3, the cache) —
    the same inputs under a different model are NOT the same computation.
    """
    canonical = json.dumps(
        {
            "model_name": spec.model_name,
            "model_version": spec.model_version,
            "features": {k: payload[k] for k in spec.features},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RiskService:
    """Orchestrates repository + feature building + model calls."""

    def __init__(
        self,
        mlflow_client: MLflowRiskClient,
        *,
        db_factory: DbFactory = open_db,
        clinic_today: date = CLINIC_TODAY,
        patients: PatientRepository | None = None,
        writer: RiskWriter | None = None,
        trends: TrendBuilder | None = None,
    ) -> None:
        self._models = mlflow_client
        self._db = db_factory
        self._today = clinic_today
        self._patients = patients or PatientRepository()
        self._writer = writer or RiskWriter()
        self._trends = trends or TrendBuilder()

    # -- reads ------------------------------------------------------------

    async def _load_record(self, patient_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        async with self._db() as db:
            record = await self._patients.fetch_record(db, patient_id)
        if record is None:
            raise PatientNotFoundError(patient_id)
        return record

    async def get_current_biomarkers(self, patient_id: str) -> BiomarkersResponse:
        """Latest biomarker snapshot for a patient (404 if unknown)."""
        demographics, biomarkers = await self._load_record(patient_id)
        if not biomarkers:
            raise IncompletePatientDataError(patient_id, ["biomarkers"])

        return BiomarkersResponse(
            patient_id=patient_id,
            name=f"{demographics['first_name']} {demographics['last_name']}",
            age_years=whole_years(
                date.fromisoformat(demographics["date_of_birth"]), self._today
            ),
            sex=demographics["sex"],
            biomarkers=BiomarkerSnapshot(**{**biomarkers, "patient_id": patient_id}),
        )

    # -- the meaty one ----------------------------------------------------

    async def get_current_risks(self, patient_id: str) -> RisksResponse:
        """Compute all five risks live, append them, and return them with trends."""
        demographics, biomarkers = await self._load_record(patient_id)
        derived = derive_features({**demographics, **biomarkers}, today=self._today)

        payloads = {
            spec.risk_code: build_payload(spec, derived, patient_id=patient_id)
            for spec in MODEL_SPECS
        }

        # The five models are independent — fire them together rather than paying
        # five sequential round trips.
        probabilities = await asyncio.gather(
            *(
                self._models.predict_proba(spec.model_name, payloads[spec.risk_code])
                for spec in MODEL_SPECS
            )
        )

        computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        hashes = {
            spec.risk_code: _inputs_hash(spec, payloads[spec.risk_code])
            for spec in MODEL_SPECS
        }

        async with self._db() as db:
            previous = await self._writer.fetch_last_inputs_hashes(db, patient_id)

            pending: list[RiskRow] = []
            for spec, probability in zip(MODEL_SPECS, probabilities, strict=True):
                digest = hashes[spec.risk_code]
                if previous.get(spec.risk_code) == digest:
                    continue  # inputs unchanged since the last stored row
                pending.append(
                    RiskRow(
                        patient_id=patient_id,
                        risk_code=spec.risk_code,
                        probability=probability,
                        risk_band=band(probability),
                        model_name=spec.model_name,
                        model_version=spec.model_version,
                        time_horizon_years=spec.time_horizon_years,
                        computed_at=computed_at,
                        inputs_json=json.dumps(
                            {
                                "inputs_hash": digest,
                                "model_name": spec.model_name,
                                "model_version": spec.model_version,
                                "clinic_today": self._today.isoformat(),
                                "features": payloads[spec.risk_code],
                                "defaults_applied": defaults_applied(spec, derived),
                            },
                            sort_keys=True,
                        ),
                    )
                )

            await self._writer.append(db, pending)
            persisted = {row.risk_code for row in pending}
            trends = await self._trends.fetch(db, patient_id)

        risks = [
            RiskResult(
                risk_code=spec.risk_code,
                probability=probability,
                risk_band=band(probability),
                model_name=spec.model_name,
                model_version=spec.model_version,
                time_horizon_years=spec.time_horizon_years,
                computed_at=computed_at,
                inputs_hash=hashes[spec.risk_code],
                persisted=spec.risk_code in persisted,
                trend_direction=direction(trends.get(spec.risk_code, [])),
            )
            for spec, probability in zip(MODEL_SPECS, probabilities, strict=True)
        ]

        return RisksResponse(
            patient_id=patient_id,
            name=f"{demographics['first_name']} {demographics['last_name']}",
            computed_at=computed_at,
            risks=risks,
            trends={
                code: [RiskTrendPoint(**point) for point in points]
                for code, points in trends.items()
            },
        )
