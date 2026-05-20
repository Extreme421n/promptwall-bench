"""add promptwall_candidate_decisions

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-18
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from app.models import PromptWallCandidateDecision

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    PromptWallCandidateDecision.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    PromptWallCandidateDecision.__table__.drop(bind=bind, checkfirst=True)
