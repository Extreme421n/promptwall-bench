"""evaluation_results: add expected_domain + risk columns

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-19

Phase E1 — store the case's declared domain and risk on each result so the
report can compute per-domain and per-risk metrics without re-loading the
eval JSONL.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _existing_columns(table: str) -> set[str]:
    return {c["name"] for c in inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    # Migration 0001 calls ``Base.metadata.create_all`` which already creates
    # the new columns from the current model definition on fresh DBs. Existing
    # DBs (at revision 0005) need a real ALTER TABLE. Check first to make this
    # migration idempotent in both situations.
    existing = _existing_columns("evaluation_results")
    if "expected_domain" not in existing or "risk" not in existing:
        with op.batch_alter_table("evaluation_results") as batch_op:
            if "expected_domain" not in existing:
                batch_op.add_column(
                    sa.Column("expected_domain", sa.String(length=40), nullable=True)
                )
            if "risk" not in existing:
                batch_op.add_column(
                    sa.Column("risk", sa.String(length=20), nullable=True)
                )


def downgrade() -> None:
    existing = _existing_columns("evaluation_results")
    if "risk" in existing or "expected_domain" in existing:
        with op.batch_alter_table("evaluation_results") as batch_op:
            if "risk" in existing:
                batch_op.drop_column("risk")
            if "expected_domain" in existing:
                batch_op.drop_column("expected_domain")
