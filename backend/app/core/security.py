"""Actor identity and access policy.

FRAMING. The brief says "all doctors can see all patients". That is a legitimate
model for a single small clinic — it is not automatically a flaw. The flaw is
that it was *implicit*: no identity, no policy object, no audit trail, and so no
way to change the policy or prove what it was. Nothing here overrides the brief.
It makes the specified behaviour explicit, configurable and audited, and the
DEFAULT (`RBAC_MODE=clinic_wide`, role `physician`) is exactly what the brief
describes.

Two dimensions, kept separate because they answer different questions:

* **Role** — what KIND of thing may this actor do? (read labs, compute risks,
  persist a result, read the audit log)
* **Scope** — WHICH patients? `clinic_wide` (every patient in the clinic, the
  brief's model) or `care_team` (only patients this actor is assigned to).

Flipping a clinic from one scope model to the other is one config value, which is
the whole point of writing it down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    PHYSICIAN = "physician"
    NURSE = "nurse"
    RESEARCHER = "researcher"
    AUDITOR = "auditor"


class Action(StrEnum):
    READ_BIOMARKERS = "read_biomarkers"
    READ_RISKS = "read_risks"
    # Separate from READ_RISKS on purpose: computing a risk to show someone is a
    # different act from writing it into the patient's permanent record.
    PERSIST_RISKS = "persist_risks"
    FIND_PATIENT = "find_patient"
    READ_AUDIT = "read_audit"


# Role -> allowed actions. Deny by default: an action absent here is refused.
POLICY: dict[Role, frozenset[Action]] = {
    Role.PHYSICIAN: frozenset(
        {
            Action.READ_BIOMARKERS,
            Action.READ_RISKS,
            Action.PERSIST_RISKS,
            Action.FIND_PATIENT,
        }
    ),
    # Nurses see everything clinical but do not write to the risk log — the
    # trend is a clinical record, and who added to it matters.
    Role.NURSE: frozenset(
        {Action.READ_BIOMARKERS, Action.READ_RISKS, Action.FIND_PATIENT}
    ),
    # Researchers get numbers, never identities. Enforced by de-identifying the
    # response (see core/phi.py), not merely by asking nicely.
    Role.RESEARCHER: frozenset({Action.READ_BIOMARKERS, Action.READ_RISKS}),
    # Auditors read the log. Deliberately no clinical access at all: the person
    # checking who looked at what should not thereby be able to look at it.
    Role.AUDITOR: frozenset({Action.READ_AUDIT}),
}

# Roles whose responses must be stripped of identifying detail.
DEIDENTIFIED_ROLES: frozenset[Role] = frozenset({Role.RESEARCHER})


@dataclass(frozen=True)
class Actor:
    """Who is asking. Built from the verified MCP token, never from user input."""

    actor_id: str
    role: Role
    clinic_id: str = "clinic-1"
    # Only consulted when RBAC_MODE=care_team.
    care_team_patients: frozenset[str] = field(default_factory=frozenset)

    @property
    def deidentified(self) -> bool:
        return self.role in DEIDENTIFIED_ROLES

    def may(self, action: Action) -> bool:
        return action in POLICY.get(self.role, frozenset())


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # lets callers write `if decision:`
        return self.allowed


def can(
    actor: Actor,
    action: Action,
    *,
    patient_id: str | None = None,
    scope_mode: str = "clinic_wide",
) -> Decision:
    """Single entry point for every access question.

    Returns a Decision rather than a bool so the reason can be written to the
    audit log — "denied" without a reason is not much of an audit trail.
    """
    if not actor.may(action):
        return Decision(False, f"role '{actor.role}' may not {action}")

    if patient_id and scope_mode == "care_team":
        if patient_id not in actor.care_team_patients:
            return Decision(
                False, f"patient {patient_id} is not on {actor.actor_id}'s care team"
            )

    return Decision(True)


# The identity used when no per-role token is presented. Keeps the assignment's
# default behaviour — a single clinic where every doctor sees every patient —
# working exactly as described, with the policy now explicit rather than absent.
DEFAULT_ACTOR = Actor(actor_id="clinic-default", role=Role.PHYSICIAN)


def actor_from_token(client_id: str | None, scopes: list[str] | None) -> Actor:
    """Build an Actor from a VERIFIED MCP token.

    The role travels as a `role:<name>` scope and the identity as the token's
    client_id. Both come from the token verifier, so a caller cannot assert a
    role by asking for one. An unrecognised or absent role falls back to the
    default physician actor, preserving the brief's behaviour.
    """
    granted = set(scopes or [])
    role = Role.PHYSICIAN
    for candidate in Role:
        if f"role:{candidate}" in granted:
            role = candidate
            break
    return Actor(actor_id=client_id or DEFAULT_ACTOR.actor_id, role=role)
