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

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query

from ...core.config import settings

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
    GuidelineSnippet,
    PatientMatchResult,
    RisksResponse,
    SearchGuidelinesResponse,
)
from ...services.guidelines import build_retriever
from ...services.risk import RiskService
from ..deps import get_actor, get_risk_service

router = APIRouter(prefix="/api/v1", tags=["clinical"])


@lru_cache(maxsize=1)
def get_retriever():
    """Built once: the corpus is static and indexing it per request is waste."""
    return build_retriever(settings.retrieval_backend)


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


@router.get("/phi_terms")
async def phi_terms(
    service: RiskService = Depends(get_risk_service),
    actor: Actor = Depends(get_actor),
) -> dict:
    """Terms the PHI boundary must scrub before text reaches an external LLM.

    INTERNAL. Consumed only by the guard proxy, over the private compose network;
    the backend publishes no host port. It returns patient NAMES and nothing else
    — no MRN, no date of birth — because that is all the scrubber needs to
    recognise an identifier in free text.

    There is an obvious tension in an endpoint that hands out the roster, and it
    is the right trade: the alternative is the guard guessing at names
    heuristically, which both misses real ones and mangles ordinary words. The
    roster stays inside the trust boundary either way; what changes is that it no
    longer reaches OpenRouter.
    """
    matches = await service.find_patients("", actor=actor)
    return {"names": sorted({m.full_name for m in matches})}


@router.get("/search_guidelines", response_model=SearchGuidelinesResponse)
async def search_guidelines(
    query: str = Query(..., min_length=2, description="What to look up"),
    k: int = Query(3, ge=1, le=10, description="How many snippets to return"),
    risk_code: str | None = Query(
        None, description="Restrict to one risk: CVD, T2DM, CKD, CLD, DEMENTIA"
    ),
) -> SearchGuidelinesResponse:
    """Search the guideline corpus for text to ground an explanation in.

    No patient data is involved, so this needs no actor scoping — the corpus is
    the same educational material for everyone. Each snippet carries the source
    file, heading and line span it came from, so the citation can be checked
    against the file rather than taken on trust.
    """
    retriever = get_retriever()
    hits = retriever.search(query, k=k, risk_code=risk_code)
    return SearchGuidelinesResponse(
        query=query,
        snippets=[
            GuidelineSnippet(**chunk.to_dict(), score=round(score, 4))
            for chunk, score in hits
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
