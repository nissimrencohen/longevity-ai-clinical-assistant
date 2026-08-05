"""Pydantic response models for the API.

These are a STARTING POINT that defines the shape the MCP tools (and the
assistant) will consume. Extend or adjust them as you implement the endpoints —
but keep responses typed and predictable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BiomarkerSnapshot(BaseModel):
    """Latest labs/vitals for a patient (subset shown; include what you need)."""

    patient_id: str
    measured_at: str
    systolic_bp: float | None = None
    diastolic_bp: float | None = None
    total_cholesterol_mgdl: float | None = None
    hdl_cholesterol_mgdl: float | None = None
    ldl_cholesterol_mgdl: float | None = None
    triglycerides_mgdl: float | None = None
    hba1c_percent: float | None = None
    fasting_glucose_mgdl: float | None = None
    egfr_ml_min_1_73m2: float | None = None
    creatinine_mgdl: float | None = None
    uacr_mg_g: float | None = None
    urine_dipstick_protein: str | None = None
    ggt_u_l: float | None = None
    alt_u_l: float | None = None
    ast_u_l: float | None = None


class BiomarkersResponse(BaseModel):
    patient_id: str
    name: str
    age_years: int
    sex: str
    biomarkers: BiomarkerSnapshot


class RiskDriver(BaseModel):
    """One feature's contribution to a risk, from the model's SHAP decomposition.

    IMPORTANT: ``contribution_log_odds`` is additive in LOG-ODDS, not in
    probability. It must never be restated as "this feature adds N% of risk" —
    that conversion is invalid and would be a confident, wrong statement in a
    clinical conversation.
    """

    feature: str = Field(examples=["egfr", "age_years"])
    label: str = Field(description="Human-readable feature name", examples=["eGFR"])
    patient_value: float | None = None
    reference_value: float | None = Field(
        default=None,
        description="The healthy-reference value this was compared against",
    )
    direction: str = Field(examples=["increases_risk", "decreases_risk"])
    contribution_log_odds: float = Field(
        description="SHAP value in log-odds. Additive in log-odds ONLY."
    )
    share_of_deviation: float | None = Field(
        default=None,
        description=(
            "Fraction of this patient's TOTAL log-odds movement away from the "
            "reference that this feature accounts for. Not a share of risk."
        ),
    )


class RiskResult(BaseModel):
    """A single freshly-computed risk."""

    risk_code: str = Field(examples=["CVD", "T2DM", "CKD", "CLD", "DEMENTIA"])
    probability: float = Field(ge=0.0, le=1.0)
    risk_band: str = Field(examples=["low", "borderline", "intermediate", "high"])
    model_name: str
    model_version: str | None = None
    time_horizon_years: int | None = None
    computed_at: str

    # Fingerprint of the exact feature payload scored (model name + version +
    # features). Ties this number to its audit row in `risks.inputs_json`.
    inputs_hash: str | None = None

    # False when the dedupe rule skipped the append because the inputs were
    # unchanged since the last stored row. The probability is still freshly
    # computed — this only reports whether the log grew.
    persisted: bool = True

    # Direction of travel vs. the previous stored point. Provided explicitly so
    # the assistant reports a trend rather than eyeballing one from the series.
    trend_direction: str | None = Field(
        default=None, examples=["worsening", "improving", "stable", "insufficient_history"]
    )

    # Where this number came from. On a cache hit `computed_at` is the ORIGINAL
    # computation time, not the time of this request — presenting an old value as
    # freshly computed is a provenance problem, not a performance one.
    source: str = Field(default="fresh", examples=["fresh", "cache"])

    # The features that moved this prediction most, largest first.
    drivers: list[RiskDriver] = Field(default_factory=list)

    # Baseline the drivers are measured against, so an explanation can always be
    # traced to the reference population it used.
    explanation_reference: str | None = Field(
        default=None, examples=["healthy-anchor-v1"]
    )


class RiskTrendPoint(BaseModel):
    computed_at: str
    probability: float
    risk_band: str


class RisksResponse(BaseModel):
    patient_id: str
    name: str
    computed_at: str
    risks: list[RiskResult]
    # Optional: prior computed values per risk_code so the assistant can show a trend.
    trends: dict[str, list[RiskTrendPoint]] = Field(default_factory=dict)
