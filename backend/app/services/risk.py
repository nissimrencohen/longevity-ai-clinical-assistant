"""Risk computation service — the core of the backend.

``get_current_risks`` end to end:

  1. Load the patient's demographics + latest biomarkers (404 if unknown).
  2. Derive the union of model inputs, then select each model's own payload.
  3. Look each payload up in the cache; call the models CONCURRENTLY for misses.
  4. Band each probability.
  5. Append to the ``risks`` log — but only when the inputs actually changed,
     which keeps the trend meaningful and makes this GET idempotent.
  6. Read the log back as a trend, with an explicit direction per risk.

The storage backend (SQLite or Postgres) and the cache (none or Redis) are both
injected, so this file contains no branching on either. Behaviour is identical
across backends and the same tests run against both.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any

from ..core.errors import IncompletePatientDataError, PatientNotFoundError
from ..db.store import ClinicalStore, RiskRow, SqliteStore, direction
from ..schemas import (
    BiomarkerSnapshot,
    BiomarkersResponse,
    RiskResult,
    RisksResponse,
    RiskTrendPoint,
)
from .banding import band
from .cache import NullCache, RiskCache, cache_key
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


def _inputs_hash(spec: ModelSpec, payload: dict[str, float]) -> str:
    """Stable fingerprint of exactly what was scored.

    Includes the model name and version, so re-registering a model with new
    coefficients correctly invalidates both the dedupe and the cache — the same
    inputs under a different model are NOT the same computation.
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
    """Orchestrates store + feature building + model calls + cache."""

    def __init__(
        self,
        mlflow_client: MLflowRiskClient,
        *,
        store: ClinicalStore | None = None,
        cache: RiskCache | None = None,
        clinic_today: date = CLINIC_TODAY,
    ) -> None:
        self._models = mlflow_client
        self._store = store or SqliteStore()
        self._cache = cache or NullCache()
        self._today = clinic_today

    # -- reads ------------------------------------------------------------

    async def _load_record(self, patient_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        record = await self._store.fetch_record(patient_id)
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
                date.fromisoformat(str(demographics["date_of_birth"])), self._today
            ),
            sex=demographics["sex"],
            biomarkers=BiomarkerSnapshot(**{**biomarkers, "patient_id": patient_id}),
        )

    # -- the meaty one ----------------------------------------------------

    async def _score(
        self, spec: ModelSpec, payload: dict[str, float], digest: str
    ) -> tuple[float, str, str | None]:
        """Return ``(probability, source, computed_at_override)``.

        A cache hit carries the ORIGINAL computation time, so the response never
        presents an hour-old number as if it were computed just now.
        """
        key = cache_key(spec.model_name, digest)
        cached = await self._cache.get(key)
        if cached and "probability" in cached:
            return float(cached["probability"]), "cache", cached.get("computed_at")

        probability = await self._models.predict_proba(spec.model_name, payload)
        return probability, "fresh", None

    async def get_current_risks(self, patient_id: str) -> RisksResponse:
        """Compute all five risks live, append them, and return them with trends."""
        demographics, biomarkers = await self._load_record(patient_id)
        derived = derive_features({**demographics, **biomarkers}, today=self._today)

        payloads = {
            spec.risk_code: build_payload(spec, derived, patient_id=patient_id)
            for spec in MODEL_SPECS
        }
        hashes = {
            spec.risk_code: _inputs_hash(spec, payloads[spec.risk_code])
            for spec in MODEL_SPECS
        }

        # The five models are independent — fire them together rather than
        # paying five sequential round trips.
        scored = await asyncio.gather(
            *(
                self._score(spec, payloads[spec.risk_code], hashes[spec.risk_code])
                for spec in MODEL_SPECS
            )
        )

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        previous = await self._store.fetch_last_inputs_hashes(patient_id)
        pending: list[RiskRow] = []
        for spec, (probability, source, _) in zip(MODEL_SPECS, scored, strict=True):
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
                    computed_at=now,
                    inputs_hash=digest,
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

        written = await self._store.append_risks(pending)
        persisted = {row.risk_code for row in written}

        # Populate the cache only after a successful computation, and store the
        # timestamp alongside so a later hit can report when it really ran.
        for spec, (probability, source, _) in zip(MODEL_SPECS, scored, strict=True):
            if source == "fresh":
                await self._cache.set(
                    cache_key(spec.model_name, hashes[spec.risk_code]),
                    {"probability": probability, "computed_at": now},
                )

        trends = await self._store.fetch_trends(patient_id)

        risks = [
            RiskResult(
                risk_code=spec.risk_code,
                probability=probability,
                risk_band=band(probability),
                model_name=spec.model_name,
                model_version=spec.model_version,
                time_horizon_years=spec.time_horizon_years,
                computed_at=computed_at or now,
                inputs_hash=hashes[spec.risk_code],
                persisted=spec.risk_code in persisted,
                source=source,
                trend_direction=direction(trends.get(spec.risk_code, [])),
            )
            for spec, (probability, source, computed_at) in zip(
                MODEL_SPECS, scored, strict=True
            )
        ]

        return RisksResponse(
            patient_id=patient_id,
            name=f"{demographics['first_name']} {demographics['last_name']}",
            computed_at=now,
            risks=risks,
            trends={
                code: [RiskTrendPoint(**point) for point in points]
                for code, points in trends.items()
            },
        )
