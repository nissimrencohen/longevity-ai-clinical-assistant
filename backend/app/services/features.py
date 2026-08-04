"""Patient record -> per-model feature payloads.

Pure and synchronous on purpose: no DB, no HTTP, no clock. Every unit here is
directly testable, which matters because this module is where "numeric
faithfulness" is won or lost.

Two things drive the design:

1. **The models own their feature lists.** ``MODEL_SPECS`` below mirrors each
   pickle's ``feature_names_in_``, and ``tests/test_features.py`` asserts the two
   agree exactly. So the pickles remain the source of truth, while the API process
   never has to load sklearn or the model files at runtime.

2. **Names in the DB are not names in the models.** ``hx_diabetes`` -> ``diabetes``,
   ``physical_activity_active`` -> ``physically_active``, and several inputs
   (``age_years``, ``bmi``, ``waist_hip_ratio``, the 0/1 flags) do not exist as
   columns at all and must be derived. See ``derive_features``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..core.errors import IncompletePatientDataError

# "Today" for this clinic, per data/DATA_DICTIONARY.md. Ages are derived against
# this FIXED date, not the wall clock: the gold probabilities in evals/cases.jsonl
# are anchored to it, so using date.today() would make every risk drift out of
# tolerance as the year advances.
CLINIC_TODAY = date(2026, 7, 9)


@dataclass(frozen=True)
class ModelSpec:
    """One risk model: what it is called, what it eats, what it reports."""

    risk_code: str
    model_name: str
    features: tuple[str, ...]
    time_horizon_years: int | None
    model_version: str = "1.0.0"


# Order is the order risks are returned in. Feature tuples mirror each model's
# feature_names_in_ exactly (name AND order) — verified by test.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        risk_code="CVD",
        model_name="prevent_cvd",
        features=(
            "age_years", "sex_male", "total_cholesterol_mgdl", "hdl_cholesterol_mgdl",
            "systolic_bp", "bp_treated", "on_statin", "diabetes", "current_smoker",
            "bmi", "egfr",
        ),
        time_horizon_years=10,
    ),
    ModelSpec(
        risk_code="T2DM",
        model_name="ada_t2dm",
        features=(
            "age_years", "sex_male", "bmi", "family_history_diabetes",
            "hypertension", "physically_active", "gestational_diabetes",
        ),
        time_horizon_years=None,
    ),
    ModelSpec(
        risk_code="CKD",
        model_name="framingham_ckd",
        features=("age_years", "diabetes", "hypertension", "proteinuria_trace_plus", "egfr"),
        time_horizon_years=10,
    ),
    ModelSpec(
        risk_code="CLD",
        model_name="clivd_cld",
        features=(
            "age_years", "sex_male", "alcohol_drinks_per_week", "waist_hip_ratio",
            "diabetes", "current_smoker", "ggt_u_l",
        ),
        time_horizon_years=15,
    ),
    ModelSpec(
        risk_code="DEMENTIA",
        model_name="caide_dementia",
        features=(
            "age_years", "sex_male", "education_years", "systolic_bp", "bmi",
            "total_cholesterol_mgdl", "physically_active",
        ),
        time_horizon_years=20,
    ),
)

SPECS_BY_RISK_CODE = {s.risk_code: s for s in MODEL_SPECS}


# The ONLY feature we are willing to fill in when the database says NULL.
#
# `gestational_diabetes` is NULL for every male patient in the dataset (P002,
# P004, P005, P008) because the question is not applicable to them — it is "N/A",
# not "unknown". The ADA model still demands the column, and sklearn raises on
# NaN, so a naive payload 500s for half the cohort. Encoding not-applicable as
# 0 ("this risk factor is absent") is the clinically correct reading and matches
# how the model was calibrated (its male reference patients use 0).
#
# Every other missing input raises instead of defaulting: silently imputing a
# lab value would let the assistant report a confident risk built on a number
# nobody measured. The substitution is recorded in inputs_json for audit.
OPTIONAL_DEFAULTS: dict[str, float] = {"gestational_diabetes": 0.0}


def whole_years(dob: date, on: date) -> int:
    """Completed years between ``dob`` and ``on`` (birthday-aware)."""
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def derive_features(
    record: Mapping[str, Any], *, today: date = CLINIC_TODAY
) -> dict[str, float | None]:
    """Build the union of every model input from a joined patient record.

    ``record`` is demographics + the latest biomarker snapshot. Returns raw values
    (``None`` preserved) — validation happens per model in ``build_payload``, so a
    patient missing a liver enzyme can still get a kidney risk.
    """

    def num(key: str) -> float | None:
        value = record.get(key)
        return None if value is None else float(value)

    height_cm = num("height_cm")
    weight_kg = num("weight_kg")
    waist_cm = num("waist_cm")
    hip_cm = num("hip_cm")

    bmi = (
        weight_kg / (height_cm / 100.0) ** 2
        if weight_kg is not None and height_cm
        else None
    )
    waist_hip_ratio = waist_cm / hip_cm if waist_cm is not None and hip_cm else None

    dipstick = record.get("urine_dipstick_protein")
    smoking = record.get("smoking_status")
    sex = record.get("sex")

    return {
        # --- derived -----------------------------------------------------
        "age_years": float(whole_years(_as_date(record["date_of_birth"]), today)),
        "bmi": bmi,
        "waist_hip_ratio": waist_hip_ratio,
        "sex_male": None if sex is None else float(sex == "male"),
        # 'former' smokers are NOT current smokers — the flag is present-tense.
        "current_smoker": None if smoking is None else float(smoking == "current"),
        # trace / 1+ / 2+ / 3+ all count as proteinuria; only 'negative' is 0.
        "proteinuria_trace_plus": (
            None if dipstick is None else float(dipstick != "negative")
        ),
        # --- renamed pass-throughs ---------------------------------------
        "bp_treated": num("on_bp_medication"),
        "diabetes": num("hx_diabetes"),
        "hypertension": num("hx_hypertension"),
        "physically_active": num("physical_activity_active"),
        "egfr": num("egfr_ml_min_1_73m2"),
        # --- straight pass-throughs --------------------------------------
        "on_statin": num("on_statin"),
        "family_history_diabetes": num("family_history_diabetes"),
        "gestational_diabetes": num("gestational_diabetes"),
        "education_years": num("education_years"),
        "alcohol_drinks_per_week": num("alcohol_drinks_per_week"),
        "systolic_bp": num("systolic_bp"),
        "total_cholesterol_mgdl": num("total_cholesterol_mgdl"),
        "hdl_cholesterol_mgdl": num("hdl_cholesterol_mgdl"),
        "ggt_u_l": num("ggt_u_l"),
    }


def build_payload(
    spec: ModelSpec,
    derived: Mapping[str, float | None],
    *,
    patient_id: str,
) -> dict[str, float]:
    """Select this model's features, applying documented defaults.

    Returns a dict in ``spec.features`` order. Raises ``IncompletePatientDataError``
    if a required input is missing and has no sanctioned default.
    """
    payload: dict[str, float] = {}
    missing: list[str] = []

    for name in spec.features:
        value = derived.get(name)
        if value is None:
            if name in OPTIONAL_DEFAULTS:
                value = OPTIONAL_DEFAULTS[name]
            else:
                missing.append(name)
                continue
        payload[name] = float(value)

    if missing:
        raise IncompletePatientDataError(patient_id, missing)
    return payload


def defaults_applied(
    spec: ModelSpec, derived: Mapping[str, float | None]
) -> dict[str, float]:
    """Which sanctioned defaults this payload relied on — recorded in inputs_json."""
    return {
        name: OPTIONAL_DEFAULTS[name]
        for name in spec.features
        if name in OPTIONAL_DEFAULTS and derived.get(name) is None
    }
