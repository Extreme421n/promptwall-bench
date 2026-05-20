"""initial schema: airline + crm + support + kb + observability

Revision ID: 0001
Revises:
Create Date: 2026-05-17

The initial migration materializes every table currently declared on
``app.models.Base.metadata``. Future schema changes should be generated via
``alembic revision --autogenerate`` and produce explicit ops.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from app.models import Base

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
