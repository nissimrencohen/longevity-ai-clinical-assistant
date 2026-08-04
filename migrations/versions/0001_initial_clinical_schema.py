"""Initial clinical schema.

Mirrors the shipped SQLite fixture (data/generate_db.py) so the two backends are
interchangeable, plus the two things Postgres makes worth having:

* ``risks.inputs_hash`` as a real column rather than a field inside the
  ``inputs_json`` blob.
* A PARTIAL unique index over ``(patient_id, model_name, inputs_hash)``. This is
  the dedupe rule — append only when the inputs changed — expressed as a
  constraint, so it becomes an atomic ``INSERT ... ON CONFLICT DO NOTHING``
  rather than a read-then-write race. Partial (``WHERE inputs_hash IS NOT NULL``)
  so the seeded history rows, which have no hash, are exempt and may repeat.

Revision ID: 0001
Revises:
Create Date: 2026-08-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "demographics",
        sa.Column("patient_id", sa.String(), primary_key=True),
        sa.Column("mrn", sa.String(), nullable=False, unique=True),
        sa.Column("first_name", sa.String(), nullable=False),
        sa.Column("last_name", sa.String(), nullable=False),
        sa.Column("date_of_birth", sa.String(), nullable=False),
        sa.Column("sex", sa.String(), nullable=False),
        sa.Column("height_cm", sa.Float(), nullable=False),
        sa.Column("weight_kg", sa.Float(), nullable=False),
        sa.Column("waist_cm", sa.Float()),
        sa.Column("hip_cm", sa.Float()),
        sa.Column("education_years", sa.Integer()),
        sa.Column("smoking_status", sa.String()),
        sa.Column("alcohol_drinks_per_week", sa.Float()),
        sa.Column("physical_activity_active", sa.Integer()),
        sa.Column("family_history_diabetes", sa.Integer()),
        sa.Column("hx_diabetes", sa.Integer()),
        sa.Column("hx_hypertension", sa.Integer()),
        # NULL means "not applicable" (male patients), not "unknown".
        sa.Column("gestational_diabetes", sa.Integer()),
        sa.Column("on_bp_medication", sa.Integer()),
        sa.Column("on_statin", sa.Integer()),
        sa.CheckConstraint("sex IN ('male','female')", name="ck_demographics_sex"),
        sa.CheckConstraint(
            "smoking_status IN ('never','former','current')",
            name="ck_demographics_smoking",
        ),
    )

    op.create_table(
        "biomarkers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "patient_id",
            sa.String(),
            sa.ForeignKey("demographics.patient_id"),
            nullable=False,
        ),
        sa.Column("measured_at", sa.String(), nullable=False),
        sa.Column("systolic_bp", sa.Float()),
        sa.Column("diastolic_bp", sa.Float()),
        sa.Column("total_cholesterol_mgdl", sa.Float()),
        sa.Column("hdl_cholesterol_mgdl", sa.Float()),
        sa.Column("ldl_cholesterol_mgdl", sa.Float()),
        sa.Column("triglycerides_mgdl", sa.Float()),
        sa.Column("hba1c_percent", sa.Float()),
        sa.Column("fasting_glucose_mgdl", sa.Float()),
        sa.Column("egfr_ml_min_1_73m2", sa.Float()),
        sa.Column("creatinine_mgdl", sa.Float()),
        sa.Column("uacr_mg_g", sa.Float()),
        sa.Column("urine_dipstick_protein", sa.String()),
        sa.Column("ggt_u_l", sa.Float()),
        sa.Column("alt_u_l", sa.Float()),
        sa.Column("ast_u_l", sa.Float()),
    )
    op.create_index("idx_biomarkers_patient", "biomarkers", ["patient_id", "measured_at"])

    op.create_table(
        "risks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "patient_id",
            sa.String(),
            sa.ForeignKey("demographics.patient_id"),
            nullable=False,
        ),
        sa.Column("risk_code", sa.String(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("risk_band", sa.String()),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("model_version", sa.String()),
        sa.Column("time_horizon_years", sa.Integer()),
        sa.Column("computed_at", sa.String(), nullable=False),
        sa.Column("inputs_json", sa.Text()),
        sa.Column("inputs_hash", sa.String(length=64)),
        sa.CheckConstraint(
            "risk_code IN ('CVD','T2DM','CKD','CLD','DEMENTIA')", name="ck_risks_code"
        ),
        sa.CheckConstraint(
            "risk_band IN ('low','borderline','intermediate','high')",
            name="ck_risks_band",
        ),
        sa.CheckConstraint(
            "probability >= 0 AND probability <= 1", name="ck_risks_probability_range"
        ),
    )
    op.create_index("idx_risks_patient", "risks", ["patient_id", "risk_code", "computed_at"])
    op.create_index(
        "uq_risks_patient_model_inputs",
        "risks",
        ["patient_id", "model_name", "inputs_hash"],
        unique=True,
        postgresql_where=sa.text("inputs_hash IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_risks_patient_model_inputs", table_name="risks")
    op.drop_index("idx_risks_patient", table_name="risks")
    op.drop_table("risks")
    op.drop_index("idx_biomarkers_patient", table_name="biomarkers")
    op.drop_table("biomarkers")
    op.drop_table("demographics")
