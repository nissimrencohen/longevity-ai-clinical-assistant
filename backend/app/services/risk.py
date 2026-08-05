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

from dataclasses import dataclass, field, replace

from ..core.config import settings
from ..core.errors import (
    AccessDeniedError,
    IncompletePatientDataError,
    PatientNotFoundError,
)
from ..core.phi import bucket_age, deidentify_name
from ..core.security import DEFAULT_ACTOR, Action, Actor, can
from ..db.store import (
    AuditEntry,
    ClinicalStore,
    PatientMatch,
    RiskRow,
    SqliteStore,
    direction,
)
from ..schemas import (
    BiomarkerSnapshot,
    BiomarkersResponse,
    RiskDriver,
    RiskResult,
    RisksResponse,
    RiskTrendPoint,
)
from .banding import band
from .cache import NullCache, RiskCache, cache_key
from .explain import build_drivers
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


@dataclass(frozen=True)
class _Scored:
    """One model's result on the way through the service."""

    probability: float
    source: str
    computed_at: str | None
    drivers: list[RiskDriver] = field(default_factory=list)
    reference_id: str | None = None
    # None when the value came FROM the cache — there is nothing to write back.
    cache_payload: dict | None = None


class RiskService:
    """Orchestrates store + feature building + model calls + cache."""

    def __init__(
        self,
        mlflow_client: MLflowRiskClient,
        *,
        store: ClinicalStore | None = None,
        cache: RiskCache | None = None,
        clinic_today: date = CLINIC_TODAY,
        explain: bool = True,
    ) -> None:
        self._models = mlflow_client
        self._store = store or SqliteStore()
        self._cache = cache or NullCache()
        self._today = clinic_today
        # Explanations are exact and closed-form for these linear models, so they
        # cost a vector subtraction and ride along in the same round trip. On by
        # default; the flag exists so a caller that genuinely does not want them
        # (or an older router) can turn them off.
        self._explain = explain

    # -- reads ------------------------------------------------------------

    async def _load_record(self, patient_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        record = await self._store.fetch_record(patient_id)
        if record is None:
            raise PatientNotFoundError(patient_id)
        return record

    # -- access control ---------------------------------------------------

    async def _authorise(
        self, actor: Actor, action: Action, patient_id: str | None = None
    ) -> None:
        """Check policy and record the decision. Raises on denial.

        Both outcomes are audited. An attempt that was refused is often more
        interesting than one that succeeded, and a log that only records
        successes cannot answer the question an auditor actually asks.
        """
        scoped = actor
        if settings.rbac_mode == "care_team" and not actor.care_team_patients:
            assigned = await self._store.fetch_care_team(actor.actor_id)
            scoped = replace(actor, care_team_patients=frozenset(assigned))

        decision = can(
            scoped, action, patient_id=patient_id, scope_mode=settings.rbac_mode
        )
        await self._audit(scoped, action, patient_id, decision)
        if not decision.allowed:
            raise AccessDeniedError(decision.reason)

    async def _audit(
        self, actor: Actor, action: Action, patient_id: str | None, decision
    ) -> None:
        if not settings.audit_enabled:
            return
        entry = AuditEntry(
            occurred_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            actor_id=actor.actor_id,
            actor_role=str(actor.role),
            action=str(action),
            patient_id=patient_id,
            decision="allow" if decision.allowed else "deny",
            reason=decision.reason or None,
            detail=f"rbac_mode={settings.rbac_mode}",
        )
        try:
            await self._store.record_audit(entry)
        except Exception:  # noqa: BLE001
            # An audit-store failure must not take down clinical care. It is
            # logged upstream by the store; the request proceeds.
            return

    # -- patient lookup ---------------------------------------------------

    async def find_patients(
        self, query: str, *, actor: Actor = DEFAULT_ACTOR
    ) -> list[PatientMatch]:
        """Resolve a name to patient ids, server-side and access-controlled.

        Replaces shipping the whole clinic roster in the system prompt: only the
        patients matching this query, and only those this actor may see, ever
        leave the backend.
        """
        await self._authorise(actor, Action.FIND_PATIENT)
        matches = await self._store.find_patients(query)

        if settings.rbac_mode == "care_team":
            assigned = actor.care_team_patients or frozenset(
                await self._store.fetch_care_team(actor.actor_id)
            )
            matches = [m for m in matches if m.patient_id in assigned]
        return matches

    async def get_current_biomarkers(
        self, patient_id: str, *, actor: Actor = DEFAULT_ACTOR
    ) -> BiomarkersResponse:
        """Latest biomarker snapshot for a patient (404 if unknown)."""
        await self._authorise(actor, Action.READ_BIOMARKERS, patient_id)
        demographics, biomarkers = await self._load_record(patient_id)
        if not biomarkers:
            raise IncompletePatientDataError(patient_id, ["biomarkers"])

        # Safe Harbor: ages over 89 are re-identifying, so they are reported as
        # a bucket rather than a true value.
        true_age = whole_years(
            date.fromisoformat(str(demographics["date_of_birth"])), self._today
        )
        age_years, age_label = bucket_age(true_age)

        name = f"{demographics['first_name']} {demographics['last_name']}"
        if actor.deidentified:
            name = deidentify_name(patient_id)

        return BiomarkersResponse(
            patient_id=patient_id,
            name=name,
            age_years=age_years,
            age_label=age_label,
            sex=demographics["sex"],
            biomarkers=BiomarkerSnapshot(**{**biomarkers, "patient_id": patient_id}),
        )

    # -- the meaty one ----------------------------------------------------

    async def _score(
        self, spec: ModelSpec, payload: dict[str, float], digest: str
    ) -> _Scored:
        """Score one model, via the cache when possible.

        A cache hit carries the ORIGINAL computation time AND its explanation.
        Caching the probability but not the drivers would mean a cached answer
        silently loses its explanation — the sort of asymmetry that turns a
        performance feature into a correctness one.
        """
        key = cache_key(spec.model_name, digest)
        cached = await self._cache.get(key)
        if cached and "probability" in cached:
            return _Scored(
                probability=float(cached["probability"]),
                source="cache",
                computed_at=cached.get("computed_at"),
                drivers=build_drivers(
                    cached.get("contributions", {}),
                    cached.get("feature_values", {}),
                    cached.get("reference_values", {}),
                ),
                reference_id=cached.get("reference_id"),
                cache_payload=None,
            )

        prediction = await self._models.predict(
            spec.model_name, payload, explain=self._explain
        )
        return _Scored(
            probability=prediction.probability,
            source="fresh",
            computed_at=None,
            drivers=build_drivers(
                prediction.contributions,
                prediction.feature_values,
                prediction.reference_values,
            ),
            reference_id=prediction.reference_id,
            cache_payload={
                "probability": prediction.probability,
                "contributions": prediction.contributions,
                "feature_values": prediction.feature_values,
                "reference_values": prediction.reference_values,
                "reference_id": prediction.reference_id,
            },
        )

    async def get_current_risks(
        self, patient_id: str, *, actor: Actor = DEFAULT_ACTOR
    ) -> RisksResponse:
        """Compute all five risks live, append them, and return them with trends."""
        await self._authorise(actor, Action.READ_RISKS, patient_id)
        # Reading a risk and WRITING it into the patient's permanent record are
        # separate acts. A nurse may see the panel; the trend is a clinical
        # record and only a physician adds to it.
        may_persist = can(
            actor, Action.PERSIST_RISKS, scope_mode=settings.rbac_mode
        ).allowed

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
        for spec, result in zip(MODEL_SPECS, scored, strict=True):
            digest = hashes[spec.risk_code]
            if previous.get(spec.risk_code) == digest:
                continue  # inputs unchanged since the last stored row
            pending.append(
                RiskRow(
                    patient_id=patient_id,
                    risk_code=spec.risk_code,
                    probability=result.probability,
                    risk_band=band(result.probability),
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
                            # Store the decomposition alongside the number it
                            # explains, so an audit can reconstruct not just what
                            # was predicted but why.
                            "drivers": [d.model_dump() for d in result.drivers],
                            "explanation_reference": result.reference_id,
                        },
                        sort_keys=True,
                    ),
                )
            )

        written = await self._store.append_risks(pending) if may_persist else []
        persisted = {row.risk_code for row in written}

        # Populate the cache only after a successful computation, and store the
        # timestamp alongside so a later hit can report when it really ran.
        for spec, result in zip(MODEL_SPECS, scored, strict=True):
            if result.cache_payload is not None:
                await self._cache.set(
                    cache_key(spec.model_name, hashes[spec.risk_code]),
                    {**result.cache_payload, "computed_at": now},
                )

        trends = await self._store.fetch_trends(patient_id)

        risks = [
            RiskResult(
                risk_code=spec.risk_code,
                probability=result.probability,
                risk_band=band(result.probability),
                model_name=spec.model_name,
                model_version=spec.model_version,
                time_horizon_years=spec.time_horizon_years,
                computed_at=result.computed_at or now,
                inputs_hash=hashes[spec.risk_code],
                persisted=spec.risk_code in persisted,
                source=result.source,
                drivers=result.drivers,
                explanation_reference=result.reference_id,
                trend_direction=direction(trends.get(spec.risk_code, [])),
            )
            for spec, result in zip(MODEL_SPECS, scored, strict=True)
        ]

        name = f"{demographics['first_name']} {demographics['last_name']}"
        if actor.deidentified:
            name = deidentify_name(patient_id)

        return RisksResponse(
            patient_id=patient_id,
            name=name,
            computed_at=now,
            risks=risks,
            trends={
                code: [RiskTrendPoint(**point) for point in points]
                for code, points in trends.items()
            },
        )
