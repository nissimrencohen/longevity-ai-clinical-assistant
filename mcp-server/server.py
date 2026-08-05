"""FastMCP server — exposes the clinical backend to the assistant as MCP tools.

Static bearer-token auth over streamable HTTP, plus the clinical tools.

The bearer token is also the IDENTITY: each token carries a client_id and a
`role:<name>` scope, which the server forwards to the backend as actor headers
after verification. A caller cannot assert a role by asking for one.

Run (from the repo root, after `uv sync`):
    uv run python mcp-server/server.py

It then listens on:  http://0.0.0.0:9000/mcp/     (note the trailing slash)
Clients must send:   Authorization: Bearer <MCP_BEARER_TOKEN>   (see repo-root .env)

Why 0.0.0.0 and port 9000: LibreChat runs in Docker and reaches this server on the
host via host.docker.internal:9000 — binding 127.0.0.1 would be unreachable from the
container. See the root GUIDE.md for the full networking + LibreChat wiring.

A note on the tool docstrings below: they are not documentation for humans, they
are the prompt. The model decides whether to call a tool, and with what arguments,
from the name, the docstring and the argument descriptions alone — so each one
states when to use it, what a patient_id looks like, and what the caller must do
with the result (report the numbers verbatim; never invent them).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.dependencies import get_access_token
from pydantic import Field

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

MCP_BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "dev-longevity-token-change-me")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001")
HOST = os.getenv("MCP_HOST", "0.0.0.0")
PORT = int(os.getenv("MCP_PORT", "9000"))

# The risks endpoint fans out to five models; give it room while still bounding
# the call, so a hung model server surfaces as an error rather than a stuck chat.
BACKEND_TIMEOUT_S = float(os.getenv("MCP_BACKEND_TIMEOUT_S", "30"))

PATIENT_ID_PATTERN = re.compile(r"^P\d{3}$")

# Static token auth — fine for a dev/take-home, NOT for production (tokens are plain
# text). Any request must present `Authorization: Bearer <MCP_BEARER_TOKEN>`.
#
# The token is also the IDENTITY. Each one carries a client_id (who) and a
# `role:<name>` scope (what they may do), both of which the verifier checks — so a
# caller cannot assert a role simply by claiming one. The default token remains a
# physician, which keeps the brief's "all doctors see all patients" behaviour
# exactly as it was; the extra tokens exist so the policy can actually be
# exercised rather than merely described.
_ROLE_TOKENS: dict[str, dict[str, Any]] = {
    MCP_BEARER_TOKEN: {
        "client_id": os.getenv("MCP_ACTOR_ID", "clinic-default"),
        "scopes": ["read", "role:physician"],
    }
}
for _role in ("nurse", "researcher", "auditor"):
    _token = os.getenv(f"MCP_TOKEN_{_role.upper()}")
    if _token:
        _ROLE_TOKENS[_token] = {
            "client_id": os.getenv(f"MCP_ACTOR_ID_{_role.upper()}", f"{_role}-1"),
            "scopes": ["read", f"role:{_role}"],
        }

verifier = StaticTokenVerifier(tokens=_ROLE_TOKENS, required_scopes=["read"])

mcp = FastMCP(name="Longevity Clinical MCP", auth=verifier)


def _actor_headers() -> dict[str, str]:
    """Identity headers derived from the VERIFIED token, for the backend.

    The MCP server is the trust boundary: it authenticates, then tells the
    backend who is asking. Nothing the model or the user types influences this —
    the values come from the token verifier, not from tool arguments.
    """
    headers = {"X-Actor-Id": "clinic-default", "X-Actor-Role": "physician"}
    try:
        token = get_access_token()
    except Exception:  # noqa: BLE001 - no request context (e.g. direct call)
        return headers
    if token is None:
        return headers

    headers["X-Actor-Id"] = getattr(token, "client_id", None) or headers["X-Actor-Id"]
    for scope in getattr(token, "scopes", None) or []:
        if scope.startswith("role:"):
            headers["X-Actor-Role"] = scope.split(":", 1)[1]
            break
    return headers


PatientId = Annotated[
    str,
    Field(
        description=(
            "Patient identifier in the form 'P001' (letter P followed by three "
            "digits). This is NOT the patient's name — if you only have a name, "
            "call find_patient first to resolve it."
        ),
        examples=["P001", "P004"],
    ),
]


def _normalise_patient_id(patient_id: str) -> str:
    """Accept sloppy casing/whitespace, reject anything that is not an ID.

    Returning a ToolError here (rather than letting a name reach the backend and
    404) gives the model an actionable correction instead of a dead end.
    """
    candidate = patient_id.strip().upper()
    if not PATIENT_ID_PATTERN.match(candidate):
        raise ToolError(
            f"{patient_id!r} is not a valid patient identifier. Expected the form "
            "'P001' (letter P plus three digits). If you were given a patient's "
            "name, you need their ID — do not guess one."
        )
    return candidate


async def _call_backend(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET a backend endpoint and translate failures into model-readable errors."""
    subject = params.get("patient_id") or params.get("query") or "the request"
    try:
        async with httpx.AsyncClient(
            base_url=BACKEND_URL, timeout=BACKEND_TIMEOUT_S
        ) as http:
            response = await http.get(path, params=params, headers=_actor_headers())
    except httpx.HTTPError as exc:
        raise ToolError(
            f"The clinical backend at {BACKEND_URL} could not be reached ({exc}). "
            "No patient data is available. Tell the user the system is unavailable; "
            "do not answer from memory."
        ) from exc

    if response.status_code == 403:
        raise ToolError(
            f"Access denied for {subject}: {_detail(response)}. You are not "
            "permitted to see this. Tell the user you do not have access; do not "
            "attempt another route to the same data."
        )
    if response.status_code == 404:
        raise ToolError(
            f"No patient with ID {subject} exists in the clinic database. "
            "Do not report any biomarkers, risks or trends for this patient, and "
            "do not substitute a different patient — say that the record was not found."
        )
    if response.status_code == 422:
        raise ToolError(
            f"Patient {subject} exists but their record is missing data required "
            f"to answer: {_detail(response)}. Report what is missing; do not estimate it."
        )
    if response.status_code == 502:
        raise ToolError(
            "The risk model server is unavailable, so risks cannot be computed right "
            "now. Do not estimate risk values — say that risk scoring is temporarily "
            "unavailable."
        )
    if response.status_code != 200:
        raise ToolError(
            f"The clinical backend returned HTTP {response.status_code} for "
            f"{subject}: {_detail(response)}"
        )

    return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))[:300]
    except ValueError:
        return response.text[:300]


@mcp.tool
def ping() -> dict:
    """Connectivity check: confirms the MCP server is reachable and authorized."""
    return {"ok": True, "backend_url": BACKEND_URL}


@mcp.tool
async def find_patient(
    name: Annotated[
        str,
        Field(
            description=(
                "Full or partial patient name, e.g. 'Maya Cohen' or just 'Cohen'. "
                "Minimum two characters."
            ),
            examples=["Maya Cohen", "Friedman"],
            min_length=2,
        ),
    ],
) -> dict:
    """Look up a patient's identifier by name.

    CALL THIS FIRST whenever the doctor names a patient rather than giving an ID.
    The other tools take a `patient_id` like 'P004'; this is how you get one.

    Returns `matches`, a list of `{patient_id, name}`. Then:
      * exactly one match  -> use that patient_id.
      * several matches    -> ask the doctor which patient they mean. Do NOT pick.
      * no matches         -> tell the doctor no such patient exists. Do NOT guess
                              an ID, and do NOT fall back to a similar name.

    Only the patients you are permitted to see are returned, and only their id and
    name — never a medical record number or date of birth.
    """
    return await _call_backend("/api/v1/find_patient", {"query": name.strip()})


@mcp.tool
async def get_current_biomarkers(patient_id: PatientId) -> dict:
    """Get a patient's most recent lab results and vital signs.

    Use this for questions about measured values — blood pressure, cholesterol,
    HbA1c, glucose, eGFR, creatinine, urine protein, liver enzymes (GGT/ALT/AST) —
    and for the patient's age and sex.

    Use `get_current_risks` instead for questions about disease RISK or PROBABILITY.

    Returns the patient's name, age, sex, and a `biomarkers` object whose values
    carry the units named in the field (e.g. `hdl_cholesterol_mgdl` is mg/dL,
    `egfr_ml_min_1_73m2` is mL/min/1.73m^2, `systolic_bp` is mmHg).

    Report these numbers exactly as returned. Never round a lab value into a
    different number, and never state a value this tool did not return.
    """
    return await _call_backend(
        "/api/v1/get_current_biomarkers",
        {"patient_id": _normalise_patient_id(patient_id)},
    )


@mcp.tool
async def get_current_risks(patient_id: PatientId) -> dict:
    """Compute a patient's five disease risks live, and return them with their trend.

    Covers cardiovascular disease (CVD), type 2 diabetes (T2DM), chronic kidney
    disease (CKD), chronic liver disease (CLD) and dementia (DEMENTIA). All five
    are always returned, so one call answers a question about any of them — call
    this once per patient, not once per disease.

    Each risk includes:
      * `probability`      — 0-1 model output for the stated time horizon
      * `risk_band`        — one of low (<0.10), borderline (0.10-0.20),
                             intermediate (0.20-0.35), high (>=0.35)
      * `time_horizon_years` — the window the probability refers to (null for the
                             T2DM screening score, which is not time-bounded)
      * `trend_direction`  — worsening / improving / stable / insufficient_history,
                             comparing against this patient's previous result
      * `drivers`          — the factors that moved this risk most, largest first

    `trends` holds the full history per risk, oldest first, for describing how a
    risk has moved over time.

    HOW TO TALK ABOUT `drivers`. Each driver has a `label` (e.g. "eGFR"), the
    patient's `patient_value`, the `reference_value` it was compared against, a
    `direction` of increases_risk or decreases_risk, and
    `contribution_log_odds`.

    `contribution_log_odds` is additive in LOG-ODDS ONLY. You must NOT convert it
    into a percentage or percentage-point change in risk. Saying "elevated BMI
    adds 12% to her risk" is FALSE, however natural it sounds. Describe drivers
    qualitatively and by rank instead:
      GOOD: "The main factors raising her kidney risk are her eGFR of 52
             (against a reference of 100), her age, and proteinuria."
      BAD:  "Her eGFR contributes 34% of her kidney risk."

    Note the direction is about which way the factor PUSHES, not whether the
    value is numerically high: a LOW eGFR increases kidney risk.

    Report the probabilities and bands exactly as returned. These are decision
    support from SURROGATE models, not a diagnosis.

    Never recommend starting, stopping or dosing a medication on the basis of
    these numbers, and never name a drug with a dose as a recommendation. A high
    risk band is evidence for a conversation, not an instruction to treat: state
    what the data shows and leave the prescribing decision to the physician.
    """
    return await _call_backend(
        "/api/v1/get_current_risks",
        {"patient_id": _normalise_patient_id(patient_id)},
    )


def main() -> None:
    mcp.run(transport="http", host=HOST, port=PORT)


if __name__ == "__main__":
    main()
