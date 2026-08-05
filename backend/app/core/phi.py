"""PHI handling: what may leave the backend, and in what form.

THE HONEST CEILING, STATED FIRST. This system cannot claim end-to-end
de-identification, and pretending otherwise would be worse than not trying. The
doctor types "What is Maya Cohen's eGFR?" into a chat box, and that message goes
to OpenRouter regardless of what these tools return. Controlling the tool output
is real and worth doing; controlling the user's own words needs an agent that
owns the inbound turn, which LibreChat's built-in agent is not.

A second limit is procurement, not engineering: OpenRouter is not a HIPAA-eligible
service and will not sign a BAA. For real PHI the LLM tier belongs behind a
covered provider or an in-VPC model. No amount of application-layer work changes
that.

WHAT IS ACTUALLY ENFORCED HERE:

1. **Minimisation.** A denylist of fields that must never cross the MCP boundary
   — MRN, date of birth, raw names of other patients. Age is derived and sent;
   the birth date it came from is not. Enforced by a test that walks every tool
   response.

2. **Safe Harbor age handling.** Ages over 89 are re-identifying (few enough
   people are 94 that a birth year plus a postcode narrows to one), so they are
   reported as "90+". None of the eight mock patients is over 89, so this is
   dormant on this dataset — which is exactly when a control is cheapest to add
   and easiest to forget.

3. **Role-based de-identification.** Researchers get numbers with a stable
   pseudonym instead of a name. The pseudonym is derived per patient so a
   researcher can still follow one subject across calls without ever learning
   who they are.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Fields that must never appear in a response leaving the backend. The risk is
# not that today's schema exposes them — it does not — but that a future field
# addition quietly does. The test walks responses against this list.
PHI_DENYLIST: frozenset[str] = frozenset(
    {
        "mrn",
        "date_of_birth",
        "dob",
        "ssn",
        "national_id",
        "address",
        "postcode",
        "zip",
        "phone",
        "email",
    }
)

# HIPAA Safe Harbor: ages above 89 must be aggregated.
SAFE_HARBOR_AGE_CAP = 89
SAFE_HARBOR_AGE_LABEL = "90+"


def bucket_age(age_years: int) -> tuple[int, str | None]:
    """Return ``(reported_age, label)``.

    Ages over the cap are reported as the cap with a "90+" label rather than the
    true value, so downstream text can say "90+" without inventing a number.
    """
    if age_years > SAFE_HARBOR_AGE_CAP:
        return SAFE_HARBOR_AGE_CAP + 1, SAFE_HARBOR_AGE_LABEL
    return age_years, None


def pseudonym(patient_id: str, *, salt: str = "longevity-research") -> str:
    """Stable, non-reversible handle for a patient.

    Deterministic so a researcher can follow one subject across calls; a digest
    so the patient id itself is not recoverable from it.
    """
    digest = hashlib.sha256(f"{salt}:{patient_id}".encode()).hexdigest()
    return f"SUBJ-{digest[:8].upper()}"


def deidentify_name(patient_id: str) -> str:
    return pseudonym(patient_id)


def scrub(payload: Any) -> Any:
    """Recursively drop denylisted keys from a response.

    A belt-and-braces pass: the response models already exclude these fields, so
    this exists to catch the case where someone adds one later.
    """
    if isinstance(payload, dict):
        return {
            key: scrub(value)
            for key, value in payload.items()
            if key.lower() not in PHI_DENYLIST
        }
    if isinstance(payload, list):
        return [scrub(item) for item in payload]
    return payload


def contains_phi(payload: Any) -> list[str]:
    """Denylisted keys present anywhere in a payload. Empty means clean."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in PHI_DENYLIST:
                    found.append(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found
