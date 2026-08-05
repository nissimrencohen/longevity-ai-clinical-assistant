"""Turn SHAP contributions into the handful of drivers a clinician would read.

Eleven log-odds numbers are a decomposition, not an explanation. This picks the
few that actually moved the prediction and attaches the context needed to read
them: what the patient's value was, what it was compared against, and which way
it pushed.

The unit discipline here is the important part. Contributions are additive in
**log-odds**, not in probability. "Elevated BMI contributes +0.42 log-odds"
is true; "elevated BMI adds 12% to her risk" is not, and would be a confident,
plausible, wrong statement in a clinical conversation. So:

* nothing in this module converts a contribution to a percentage of risk;
* ``share_of_deviation`` is a share of the TOTAL LOG-ODDS MOVEMENT away from the
  reference — explicitly named and documented as such, never as risk;
* the MCP tool docstring tells the model the same thing, and an eval case checks
  the assistant does not make the conversion anyway.
"""

from __future__ import annotations

from ..schemas import RiskDriver

# Human labels for the model feature names. The assistant sees these, and
# "waist_hip_ratio" reads badly in a sentence.
FEATURE_LABELS: dict[str, str] = {
    "age_years": "age",
    "sex_male": "male sex",
    "bmi": "BMI",
    "waist_hip_ratio": "waist-hip ratio",
    "systolic_bp": "systolic blood pressure",
    "total_cholesterol_mgdl": "total cholesterol",
    "hdl_cholesterol_mgdl": "HDL cholesterol",
    "egfr": "eGFR",
    "ggt_u_l": "GGT",
    "education_years": "years of education",
    "alcohol_drinks_per_week": "alcohol intake",
    "current_smoker": "current smoking",
    "diabetes": "diabetes",
    "hypertension": "hypertension",
    "proteinuria_trace_plus": "proteinuria",
    "bp_treated": "blood-pressure treatment",
    "on_statin": "statin therapy",
    "physically_active": "physical activity",
    "family_history_diabetes": "family history of diabetes",
    "gestational_diabetes": "gestational diabetes",
}

# Below this the contribution is noise relative to the others and listing it
# implies a precision the surrogate models do not have.
MIN_ABS_CONTRIBUTION = 0.01


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def build_drivers(
    contributions: dict[str, float],
    feature_values: dict[str, float],
    reference_values: dict[str, float],
    *,
    top_k: int = 3,
) -> list[RiskDriver]:
    """The ``top_k`` features that moved this prediction most, largest first."""
    if not contributions:
        return []

    total_movement = sum(abs(value) for value in contributions.values())
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

    drivers: list[RiskDriver] = []
    for feature, contribution in ranked[:top_k]:
        if abs(contribution) < MIN_ABS_CONTRIBUTION:
            continue
        drivers.append(
            RiskDriver(
                feature=feature,
                label=label_for(feature),
                patient_value=feature_values.get(feature),
                reference_value=reference_values.get(feature),
                # Positive contributions push risk UP. Note this is about the
                # direction of the push, not whether the value is high or low:
                # a LOW eGFR raises kidney risk because its coefficient is
                # negative.
                direction="increases_risk" if contribution > 0 else "decreases_risk",
                contribution_log_odds=round(contribution, 4),
                share_of_deviation=(
                    round(abs(contribution) / total_movement, 4)
                    if total_movement > 0
                    else None
                ),
            )
        )
    return drivers
