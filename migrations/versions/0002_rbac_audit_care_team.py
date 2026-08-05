"""RBAC support: care_team assignments and the append-only audit log.

Both tables are inert under the default policy (`RBAC_MODE=clinic_wide`, which is
the brief's "all doctors see all patients") except that the audit log is written
on every access regardless of mode — recording who looked at what is not
conditional on restricting who may.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-05
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "care_team",
        sa.Column("actor_id", sa.String(), primary_key=True),
        sa.Column(
            "patient_id",
            sa.String(),
            sa.ForeignKey("demographics.patient_id"),
            primary_key=True,
        ),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("occurred_at", sa.String(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("actor_role", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("patient_id", sa.String()),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("reason", sa.String()),
        sa.Column("detail", sa.Text()),
    )
    op.create_index("idx_audit_actor", "audit_log", ["actor_id", "occurred_at"])
    op.create_index("idx_audit_patient", "audit_log", ["patient_id", "occurred_at"])


def downgrade() -> None:
    op.drop_index("idx_audit_patient", table_name="audit_log")
    op.drop_index("idx_audit_actor", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_table("care_team")
