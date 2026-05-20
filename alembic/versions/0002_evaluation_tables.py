"""add evaluation_runs and evaluation_results

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-18
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from app.models import EvaluationResult, EvaluationRun

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    EvaluationRun.__table__.create(bind=bind, checkfirst=True)
    EvaluationResult.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    EvaluationResult.__table__.drop(bind=bind, checkfirst=True)
    EvaluationRun.__table__.drop(bind=bind, checkfirst=True)
