"""Feature-building tests — where numeric faithfulness is actually decided.

The headline test is ``test_model_specs_match_pickles``: it asserts that the
backend's declarative ``MODEL_SPECS`` still matches every pickle's
``feature_names_in_``, name for name and in order. That is what lets the API
process avoid loading sklearn at runtime while keeping the models as the single
source of truth — if someone regenerates the models with a different feature set,
this fails loudly instead of the payload silently going wrong.
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import pytest

from backend.app.core.errors import IncompletePatientDataError
from backend.app.services.features import (
    CLINIC_TODAY,
    MODEL_SPECS,
    SPECS_BY_RISK_CODE,
    build_payload,
    defaults_applied,
    derive_features,
    whole_years,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# A male patient (P004 Avraham Friedman) — note gestational_diabetes is NULL.
P004 = {
    "patient_id": "P004",
    "first_name": "Avraham",
    "last_name": "Friedman",
    "date_of_birth": "1954-02-18",
    "sex": "male",
    "height_cm": 172,
    "weight_kg": 85,
    "waist_cm": 102,
    "hip_cm": 100,
    "education_years": 16,
    "smoking_status": "former",
    "alcohol_drinks_per_week": 4,
    "physical_activity_active": 0,
    "family_history_diabetes": 1,
    "hx_diabetes": 1,
    "hx_hypertension": 1,
    "gestational_diabetes": None,
    "on_bp_medication": 1,
    "on_statin": 1,
    "systolic_bp": 150,
    "total_cholesterol_mgdl": 190,
    "hdl_cholesterol_mgdl": 40,
    "egfr_ml_min_1_73m2": 52,
    "urine_dipstick_protein": "1+",
    "ggt_u_l": 40,
}


def test_model_specs_match_pickles() -> None:
    """MODEL_SPECS must mirror each model's feature_names_in_ exactly (names + order)."""
    for spec in MODEL_SPECS:
        with (MODELS_DIR / f"{spec.model_name}.pkl").open("rb") as fh:
            model = pickle.load(fh)
        assert spec.features == tuple(model.feature_names_in_), (
            f"{spec.model_name}: spec {spec.features} != pickle "
            f"{tuple(model.feature_names_in_)}"
        )
        assert spec.risk_code == model.metadata_["risk_code"]
        assert spec.model_version == model.metadata_["model_version"]
        assert spec.time_horizon_years == model.metadata_["time_horizon_years"]


def test_all_five_risk_codes_present() -> None:
    assert set(SPECS_BY_RISK_CODE) == {"CVD", "T2DM", "CKD", "CLD", "DEMENTIA"}


@pytest.mark.parametrize(
    ("dob", "expected"),
    [
        ("1954-02-18", 72),  # birthday already passed in 2026
        ("1992-03-14", 34),  # P001 Maya Cohen
        ("1958-01-22", 68),  # P002 David Levi
        ("2026-07-09", 0),   # born on clinic day
        ("2026-07-10", -1),  # day after — guards the birthday comparison
    ],
)
def test_whole_years(dob: str, expected: int) -> None:
    assert whole_years(date.fromisoformat(dob), CLINIC_TODAY) == expected


def test_age_is_anchored_to_clinic_today_not_wall_clock() -> None:
    """Ages must not drift as real time passes — the eval tolerances depend on it."""
    derived = derive_features(P004)
    assert derived["age_years"] == 72.0
    # Same record scored a year later still yields 72 under the fixed anchor.
    assert derive_features(P004, today=CLINIC_TODAY)["age_years"] == 72.0


def test_derived_quantities() -> None:
    derived = derive_features(P004)
    assert derived["bmi"] == pytest.approx(85 / 1.72**2)
    assert derived["waist_hip_ratio"] == pytest.approx(102 / 100)
    assert derived["sex_male"] == 1.0
    assert derived["bp_treated"] == 1.0
    assert derived["diabetes"] == 1.0
    assert derived["hypertension"] == 1.0
    assert derived["physically_active"] == 0.0
    assert derived["egfr"] == 52.0


def test_former_smoker_is_not_a_current_smoker() -> None:
    """`current_smoker` is present-tense — 'former' must encode as 0."""
    assert derive_features({**P004, "smoking_status": "former"})["current_smoker"] == 0.0
    assert derive_features({**P004, "smoking_status": "never"})["current_smoker"] == 0.0
    assert derive_features({**P004, "smoking_status": "current"})["current_smoker"] == 1.0


@pytest.mark.parametrize(
    ("dipstick", "expected"),
    [("negative", 0.0), ("trace", 1.0), ("1+", 1.0), ("2+", 1.0), ("3+", 1.0)],
)
def test_proteinuria_flag(dipstick: str, expected: float) -> None:
    """Only 'negative' is 0 — 'trace' and above all count as proteinuria."""
    derived = derive_features({**P004, "urine_dipstick_protein": dipstick})
    assert derived["proteinuria_trace_plus"] == expected


# ---------------------------------------------------------------------------
# The NULL gestational_diabetes regression (P002, P004, P005, P008 are all NULL).
# ---------------------------------------------------------------------------


def test_null_gestational_diabetes_is_coalesced_for_males() -> None:
    """NULL means 'not applicable', not 'unknown' — it must become 0, not NaN.

    Without this, the ADA T2DM model receives NaN and sklearn raises for exactly
    the four male patients in the dataset.
    """
    derived = derive_features(P004)
    assert derived["gestational_diabetes"] is None  # raw value preserved

    spec = SPECS_BY_RISK_CODE["T2DM"]
    payload = build_payload(spec, derived, patient_id="P004")
    assert payload["gestational_diabetes"] == 0.0
    assert not any(v != v for v in payload.values()), "payload must contain no NaN"


def test_applied_defaults_are_recorded_for_audit() -> None:
    """The substitution must be visible in inputs_json, not silent."""
    derived = derive_features(P004)
    assert defaults_applied(SPECS_BY_RISK_CODE["T2DM"], derived) == {
        "gestational_diabetes": 0.0
    }
    # A female patient with a real recorded value relies on no defaults.
    female = derive_features({**P004, "sex": "female", "gestational_diabetes": 1})
    assert defaults_applied(SPECS_BY_RISK_CODE["T2DM"], female) == {}


def test_missing_lab_refuses_to_score_rather_than_imputing() -> None:
    """A missing eGFR must raise, not quietly become 0 — that would invert CKD risk."""
    derived = derive_features({**P004, "egfr_ml_min_1_73m2": None})
    with pytest.raises(IncompletePatientDataError) as excinfo:
        build_payload(SPECS_BY_RISK_CODE["CKD"], derived, patient_id="P004")
    assert "egfr" in excinfo.value.missing


def test_payload_is_in_model_feature_order() -> None:
    """Order matters: sklearn matches on position once names are stripped."""
    derived = derive_features(P004)
    for spec in MODEL_SPECS:
        payload = build_payload(spec, derived, patient_id="P004")
        assert tuple(payload) == spec.features
