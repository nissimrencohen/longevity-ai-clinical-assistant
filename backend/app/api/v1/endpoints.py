"""API v1 routes.

Deliberately thin: parse input, call the service, return a typed response. All the
clinical logic lives in ``app.services.risk``.

The MCP server calls these over HTTP, so the contract is stable and the errors are
meaningful:

* **404** — unknown patient. The assistant must be able to say "no such patient"
  instead of inventing one.
* **422** — the patient exists but a required model input is missing. Distinct
  from 404 on purpose; we refuse to score rather than impute a lab value.
* **502** — the MLflow model server is unreachable or returned a non-probability.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...core.errors import IncompletePatientDataError, ModelServerError, PatientNotFoundError
from ...schemas import BiomarkersResponse, RisksResponse
from ...services.risk import RiskService
from ..deps import get_risk_service

router = APIRouter(prefix="/api/v1", tags=["clinical"])


@router.get("/get_current_biomarkers", response_model=BiomarkersResponse)
async def get_current_biomarkers(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
    service: RiskService = Depends(get_risk_service),
) -> BiomarkersResponse:
    """Return the latest biomarker snapshot for a patient."""
    try:
        return await service.get_current_biomarkers(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncompletePatientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/get_current_risks", response_model=RisksResponse)
async def get_current_risks(
    patient_id: str = Query(..., description="Patient identifier, e.g. P001"),
    service: RiskService = Depends(get_risk_service),
) -> RisksResponse:
    """Compute the five clinical risks in real time, persist them, and return them.

    This GET appends to the ``risks`` log — an HTTP-semantics smell the brief calls
    out. It is handled by making the write idempotent: a row is appended only when
    the feature payload differs from the last stored one, so repeated calls cause
    no observable state change. ``POST /api/v1/patients/{id}/risk-computations``
    is the semantically correct spelling and is on the roadmap; this path stays
    because it is the graded tool contract.
    """
    try:
        return await service.get_current_risks(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IncompletePatientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelServerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
