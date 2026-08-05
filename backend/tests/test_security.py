"""RBAC, PHI minimisation, and find_patient.

The brief says "all doctors can see all patients". Nothing here overrides that —
the DEFAULT configuration behaves exactly as the brief describes, and the first
test asserts it. What is new is that the policy is explicit, configurable and
audited, so it can be changed and proved rather than merely assumed.
"""

from __future__ import annotations

import sqlite3

import pytest

from backend.app.core import phi
from backend.app.core.config import settings
from backend.app.core.errors import AccessDeniedError
from backend.app.core.security import (
    DEFAULT_ACTOR,
    Action,
    Actor,
    Role,
    actor_from_token,
    can,
)

PHYSICIAN = Actor(actor_id="dr-cohen", role=Role.PHYSICIAN)
NURSE = Actor(actor_id="nurse-1", role=Role.NURSE)
RESEARCHER = Actor(actor_id="res-1", role=Role.RESEARCHER)
AUDITOR = Actor(actor_id="aud-1", role=Role.AUDITOR)


# ---------------------------------------------------------------------------
# The brief's model is the default
# ---------------------------------------------------------------------------


def test_default_actor_is_a_clinic_wide_physician() -> None:
    """The assignment's behaviour, unchanged — just now written down."""
    assert DEFAULT_ACTOR.role is Role.PHYSICIAN
    assert settings.rbac_mode == "clinic_wide"
    for patient in ("P001", "P004", "P008"):
        assert can(DEFAULT_ACTOR, Action.READ_BIOMARKERS, patient_id=patient)
        assert can(DEFAULT_ACTOR, Action.READ_RISKS, patient_id=patient)


# ---------------------------------------------------------------------------
# Role matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("actor", "action", "allowed"),
    [
        (PHYSICIAN, Action.READ_BIOMARKERS, True),
        (PHYSICIAN, Action.PERSIST_RISKS, True),
        (PHYSICIAN, Action.READ_AUDIT, False),
        # A nurse reads everything clinical but does not write to the risk log.
        (NURSE, Action.READ_BIOMARKERS, True),
        (NURSE, Action.READ_RISKS, True),
        (NURSE, Action.PERSIST_RISKS, False),
        # A researcher gets numbers, never the audit trail or write access.
        (RESEARCHER, Action.READ_RISKS, True),
        (RESEARCHER, Action.PERSIST_RISKS, False),
        (RESEARCHER, Action.FIND_PATIENT, False),
        # The person checking who looked at what should not thereby be able to look.
        (AUDITOR, Action.READ_AUDIT, True),
        (AUDITOR, Action.READ_BIOMARKERS, False),
        (AUDITOR, Action.READ_RISKS, False),
    ],
)
def test_role_matrix(actor: Actor, action: Action, allowed: bool) -> None:
    assert bool(can(actor, action)) is allowed


def test_denial_carries_a_reason() -> None:
    """"Denied" with no reason is not much of an audit trail."""
    decision = can(AUDITOR, Action.READ_BIOMARKERS)
    assert not decision.allowed
    assert "auditor" in decision.reason


def test_role_cannot_be_asserted_by_the_caller() -> None:
    """Role comes from the verified token's scopes, not from anything typed."""
    assert actor_from_token("dr-x", ["read", "role:nurse"]).role is Role.NURSE
    # No role scope, or a bogus one, falls back to the default rather than up.
    assert actor_from_token("dr-x", ["read"]).role is Role.PHYSICIAN
    assert actor_from_token("dr-x", ["read", "role:admin"]).role is Role.PHYSICIAN


# ---------------------------------------------------------------------------
# Scope: care_team mode
# ---------------------------------------------------------------------------


def test_care_team_mode_restricts_by_patient() -> None:
    scoped = Actor(
        actor_id="dr-cohen",
        role=Role.PHYSICIAN,
        care_team_patients=frozenset({"P001", "P002"}),
    )
    assert can(scoped, Action.READ_RISKS, patient_id="P001", scope_mode="care_team")

    denied = can(scoped, Action.READ_RISKS, patient_id="P004", scope_mode="care_team")
    assert not denied.allowed
    assert "care team" in denied.reason


def test_clinic_wide_ignores_care_team_assignments() -> None:
    """One config value is the difference between the two clinic models."""
    scoped = Actor(
        actor_id="dr-cohen", role=Role.PHYSICIAN, care_team_patients=frozenset({"P001"})
    )
    assert can(scoped, Action.READ_RISKS, patient_id="P004", scope_mode="clinic_wide")


# ---------------------------------------------------------------------------
# PHI
# ---------------------------------------------------------------------------


def test_safe_harbor_buckets_ages_over_89() -> None:
    """Few enough people are 94 that an exact age is re-identifying.

    Dormant on this dataset — nobody here is over 89 — which is exactly when a
    control is cheapest to add and easiest to forget.
    """
    assert phi.bucket_age(72) == (72, None)
    assert phi.bucket_age(89) == (89, None)
    assert phi.bucket_age(94) == (90, "90+")


def test_pseudonym_is_stable_and_not_reversible() -> None:
    first = phi.pseudonym("P004")
    assert first == phi.pseudonym("P004"), "a researcher must be able to follow a subject"
    assert first != phi.pseudonym("P005")
    assert "P004" not in first


def test_scrub_removes_denylisted_fields() -> None:
    payload = {"patient_id": "P004", "mrn": "MRN-1004", "nested": {"date_of_birth": "1954-02-18"}}
    cleaned = phi.scrub(payload)

    assert cleaned == {"patient_id": "P004", "nested": {}}
    assert phi.contains_phi(cleaned) == []
    assert set(phi.contains_phi(payload)) == {"mrn", "date_of_birth"}


# ---------------------------------------------------------------------------
# End to end through the service
# ---------------------------------------------------------------------------


async def test_mrn_and_dob_never_cross_the_boundary(risk_service) -> None:
    """The one PHI guarantee this system can actually make and prove.

    The schema excludes these today; this asserts it stays true as fields get
    added, which is the failure mode that matters.
    """
    for patient in ("P001", "P004"):
        biomarkers = await risk_service.get_current_biomarkers(patient)
        risks = await risk_service.get_current_risks(patient)
        assert phi.contains_phi(biomarkers.model_dump()) == []
        assert phi.contains_phi(risks.model_dump()) == []


async def test_researcher_sees_numbers_but_not_names(risk_service) -> None:
    response = await risk_service.get_current_risks("P004", actor=RESEARCHER)

    assert "Avraham" not in response.name and "Friedman" not in response.name
    assert response.name == phi.pseudonym("P004")
    # The clinical numbers are untouched — de-identification, not degradation.
    ckd = next(r for r in response.risks if r.risk_code == "CKD")
    assert ckd.probability == pytest.approx(0.50, abs=1e-6)


async def test_physician_sees_the_real_name(risk_service) -> None:
    response = await risk_service.get_current_risks("P004")
    assert response.name == "Avraham Friedman"


async def test_auditor_is_refused_clinical_data(risk_service) -> None:
    with pytest.raises(AccessDeniedError):
        await risk_service.get_current_risks("P004", actor=AUDITOR)


async def test_nurse_reads_risks_but_does_not_write_the_trend(risk_service) -> None:
    """Reading a risk and adding it to the record are different acts."""

    def stored() -> int:
        con = sqlite3.connect(settings.patient_db_path)
        try:
            return con.execute(
                "SELECT COUNT(*) FROM risks WHERE patient_id='P006' "
                "AND inputs_json IS NOT NULL"
            ).fetchone()[0]
        finally:
            con.close()

    before = stored()
    response = await risk_service.get_current_risks("P006", actor=NURSE)

    assert len(response.risks) == 5, "a nurse still sees the panel"
    assert all(not r.persisted for r in response.risks)
    assert stored() == before, "no rows were appended"


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def _audit_rows(patient_id: str | None = None) -> list[tuple]:
    con = sqlite3.connect(settings.patient_db_path)
    try:
        if patient_id:
            return con.execute(
                "SELECT actor_id, actor_role, action, decision, reason FROM audit_log "
                "WHERE patient_id = ? ORDER BY id",
                (patient_id,),
            ).fetchall()
        return con.execute(
            "SELECT actor_id, actor_role, action, decision, reason FROM audit_log "
            "ORDER BY id"
        ).fetchall()
    finally:
        con.close()


async def test_successful_access_is_audited(risk_service) -> None:
    await risk_service.get_current_biomarkers("P002", actor=PHYSICIAN)
    rows = _audit_rows("P002")

    assert rows, "no audit row written"
    actor_id, role, action, decision, _reason = rows[-1]
    assert (actor_id, role, decision) == ("dr-cohen", "physician", "allow")
    assert action == "read_biomarkers"


async def test_denied_access_is_audited_too(risk_service) -> None:
    """A refused attempt is often more interesting than a successful one."""
    with pytest.raises(AccessDeniedError):
        await risk_service.get_current_biomarkers("P002", actor=AUDITOR)

    denials = [r for r in _audit_rows("P002") if r[3] == "deny"]
    assert denials, "a denial must leave a trace"
    assert denials[-1][1] == "auditor"
    assert "may not" in denials[-1][4]


# ---------------------------------------------------------------------------
# find_patient
# ---------------------------------------------------------------------------


async def test_find_patient_resolves_a_full_name(risk_service) -> None:
    matches = await risk_service.find_patients("Maya Cohen")
    assert [(m.patient_id, m.full_name) for m in matches] == [("P001", "Maya Cohen")]


async def test_find_patient_matches_a_surname(risk_service) -> None:
    matches = await risk_service.find_patients("Friedman")
    assert [m.patient_id for m in matches] == ["P004"]


async def test_find_patient_is_case_insensitive_and_partial(risk_service) -> None:
    assert [m.patient_id for m in await risk_service.find_patients("mizrahi")] == ["P003"]
    assert [m.patient_id for m in await risk_service.find_patients("kat")] == ["P005"]


async def test_find_patient_returns_nothing_for_an_unknown_name(risk_service) -> None:
    """The nearest-name substitution failure mode, closed at the source.

    "Miriam Cohen" does not exist. The tool returns no matches rather than the
    only other Cohen, so the model has nothing plausible to latch onto.
    """
    assert await risk_service.find_patients("Miriam Cohen") == []


async def test_find_patient_carries_no_phi(risk_service) -> None:
    matches = await risk_service.find_patients("Cohen")
    for match in matches:
        payload = {"patient_id": match.patient_id, "name": match.full_name}
        assert phi.contains_phi(payload) == []


async def test_researcher_may_not_look_patients_up_by_name(risk_service) -> None:
    """De-identification would be pointless if the same actor could map ids to names."""
    with pytest.raises(AccessDeniedError):
        await risk_service.find_patients("Cohen", actor=RESEARCHER)
