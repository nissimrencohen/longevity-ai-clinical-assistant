"""SQLAlchemy schema for the Postgres backend.

Mirrors the shipped SQLite schema in ``data/generate_db.py`` column for column,
so the two backends are interchangeable and the seed script is a straight copy.
``data/generate_db.py`` remains the canonical definition of the fixture and is
deliberately untouched.

Two additions Postgres makes worthwhile:

* ``risks.inputs_hash`` as a real column. Under SQLite the hash lives inside the
  ``inputs_json`` blob, because adding a column would mean editing the provided
  generator. Here it is indexed and constrained.

* A unique index over ``(patient_id, model_name, inputs_hash)``. That turns the
  dedupe rule — append only when the inputs changed — from a read-then-write
  race into a single atomic ``INSERT ... ON CONFLICT DO NOTHING``. This is the
  concrete reason to prefer Postgres here, more than raw concurrency: under
  SQLite the check and the insert are two statements with a window between them.
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

RISK_CODES = ("CVD", "T2DM", "CKD", "CLD", "DEMENTIA")
BANDS = ("low", "borderline", "intermediate", "high")

demographics = Table(
    "demographics",
    metadata,
    Column("patient_id", String, primary_key=True),
    Column("mrn", String, unique=True, nullable=False),
    Column("first_name", String, nullable=False),
    Column("last_name", String, nullable=False),
    Column("date_of_birth", String, nullable=False),
    Column("sex", String, nullable=False),
    Column("height_cm", Float, nullable=False),
    Column("weight_kg", Float, nullable=False),
    Column("waist_cm", Float),
    Column("hip_cm", Float),
    Column("education_years", Integer),
    Column("smoking_status", String),
    Column("alcohol_drinks_per_week", Float),
    Column("physical_activity_active", Integer),
    Column("family_history_diabetes", Integer),
    Column("hx_diabetes", Integer),
    Column("hx_hypertension", Integer),
    # Nullable on purpose: NULL for male patients means "not applicable", and
    # the feature builder encodes that as 0 with the substitution recorded for
    # audit. See backend/app/services/features.py.
    Column("gestational_diabetes", Integer),
    Column("on_bp_medication", Integer),
    Column("on_statin", Integer),
    CheckConstraint("sex IN ('male','female')", name="ck_demographics_sex"),
    CheckConstraint(
        "smoking_status IN ('never','former','current')",
        name="ck_demographics_smoking",
    ),
)

biomarkers = Table(
    "biomarkers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "patient_id",
        String,
        ForeignKey("demographics.patient_id"),
        nullable=False,
    ),
    Column("measured_at", String, nullable=False),
    Column("systolic_bp", Float),
    Column("diastolic_bp", Float),
    Column("total_cholesterol_mgdl", Float),
    Column("hdl_cholesterol_mgdl", Float),
    Column("ldl_cholesterol_mgdl", Float),
    Column("triglycerides_mgdl", Float),
    Column("hba1c_percent", Float),
    Column("fasting_glucose_mgdl", Float),
    Column("egfr_ml_min_1_73m2", Float),
    Column("creatinine_mgdl", Float),
    Column("uacr_mg_g", Float),
    Column("urine_dipstick_protein", String),
    Column("ggt_u_l", Float),
    Column("alt_u_l", Float),
    Column("ast_u_l", Float),
    Index("idx_biomarkers_patient", "patient_id", "measured_at"),
)

risks = Table(
    "risks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "patient_id",
        String,
        ForeignKey("demographics.patient_id"),
        nullable=False,
    ),
    Column("risk_code", String, nullable=False),
    Column("probability", Float, nullable=False),
    Column("risk_band", String),
    Column("model_name", String, nullable=False),
    Column("model_version", String),
    Column("time_horizon_years", Integer),
    Column("computed_at", String, nullable=False),
    Column("inputs_json", Text),
    # Promoted out of inputs_json so it can carry a real constraint.
    Column("inputs_hash", String(64)),
    CheckConstraint(
        "risk_code IN ('CVD','T2DM','CKD','CLD','DEMENTIA')",
        name="ck_risks_code",
    ),
    CheckConstraint(
        "risk_band IN ('low','borderline','intermediate','high')",
        name="ck_risks_band",
    ),
    CheckConstraint(
        "probability >= 0 AND probability <= 1",
        name="ck_risks_probability_range",
    ),
    Index("idx_risks_patient", "patient_id", "risk_code", "computed_at"),
    # The dedupe rule as a database constraint. Partial, so the seeded history
    # rows (inputs_hash NULL) are exempt and can repeat freely.
    Index(
        "uq_risks_patient_model_inputs",
        "patient_id",
        "model_name",
        "inputs_hash",
        unique=True,
        postgresql_where=Column("inputs_hash").isnot(None),
    ),
)
