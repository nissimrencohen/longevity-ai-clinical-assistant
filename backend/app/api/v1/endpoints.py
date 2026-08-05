"""API v1 routes.

Deliberately thin: identify the caller, parse input, call the service, return a
typed response. All the clinical logic lives in ``app.services.risk`` and all the
policy in ``app.core.security``.

The MCP server calls these over HTTP, so the contract is stable and the errors are
meaningful:

* **403** — the caller is known but not permitted. Distinct from 404 on purpose:
  answering "no such patient" to an unauthorised caller would be a small lie, and
  the denial is recorded in the audit log either way.
* **404** — unknown patient. The assistant must be able to say "no such patient"
  instead of inventing one.
* **422** — the patient exists but a required model input is missing. We refuse
  to score rather than impute a lab value.
* **502** — the MLflow model server is unreachable or returned a non-probability.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...core.errors import (
    AccessDeniedError,
    IncompletePatientDataError,
    ModelServerError,
    PatientNotFoundError,
)
from ...core.security import Actor
from ...schemas import (
    BiomarkersResponse,
    FindPatientResponse,
    PatientMatchResult,
    RisksResponse,
)
from ...services.risk import RiskService
from ..deps import get_actor, get_risk_service

router = APIRouter(prefix="/api/v1", tags=["clinical"])


@router.get("/find_patient", response_model=FindPatientResponse)
async def find_patient(
    query: str = Query(..., min_length=2, description="Full or partial patient name"),
    service: RiskService = Depends(get_risk_service),
    actor: Actor = Depends(get_actor),
) -> FindPatientResponse:
    """Resolve a patient name to an identifier.

    Exists so the clinic roster does not have to live in the assistant's system
    prompt, where it would be sent to the external model on every single turn.
    Only patients matching this query, and only those this caller may see, leave
    the backend.
    """
    try:
        matches = await service.find_patients(query, actor=actor)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return FindPatientResponse(
        query=query,
        matches=[
            PatientMatchResult(patient_id=m.patient_id, name=m.full_name)
            for m in matches
        ],
    )


@router.get("/get_current_biomarkers", response_model=BiomarkersResponse)
async def get_current_biomarkers(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
    service: RiskService = Depends(get_risk_service),
    actor: Actor = Depends(get_actor),
) -> BiomarkersResponse:
    """Return the latest biomarker snapshot for a patient."""
    try:
        return await service.get_current_biomarkers(patient_id, actor=actor)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncompletePatientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/get_current_risks", response_model=RisksResponse)
async def get_current_risks(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
    service: RiskService = Depends(get_risk_service),
    actor: Actor = Depends(get_actor),
) -> RisksResponse:
    """Compute the five clinical risks in real time, persist them, and return them.

    This GET appends to the ``risks`` log — an HTTP-semantics smell the brief calls
    out. It is handled by making the write idempotent: a row is appended only when
    the feature payload differs from the last stored one, so repeated calls cause
    no observable state change. ``POST /api/v1/patients/{id}/risk-computations``
    is the semantically correct spelling and is on the roadmap; this path stays
    because it is the graded tool contract.

    Roles without ``persist_risks`` (nurse, researcher) still receive the computed
    panel; only the append is withheld, because the trend is a clinical record.
    """
    try:
        return await service.get_current_risks(patient_id, actor=actor)
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncompletePatientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
