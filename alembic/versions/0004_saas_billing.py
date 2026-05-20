"""add saas/billing domain

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-19
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from app.models import (
    ApiUsageDaily,
    CustomerOrganization,
    Invoice,
    InvoiceItem,
    Organization,
    OverageCharge,
    Plan,
    SeatAllocation,
    Subscription,
    UsageEvent,
)

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Creation order honours FK dependencies. Drop order is the reverse.
_CREATE_ORDER = (
    Organization,
    Plan,
    CustomerOrganization,  # FK -> customers, organizations
    Subscription,          # FK -> organizations, plans
    Invoice,               # FK -> organizations, subscriptions
    InvoiceItem,           # FK -> invoices
    UsageEvent,            # FK -> organizations
    ApiUsageDaily,         # FK -> organizations
    SeatAllocation,        # FK -> organizations
    OverageCharge,         # FK -> invoices, organizations
)


def upgrade() -> None:
    bind = op.get_bind()
    for model in _CREATE_ORDER:
        model.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for model in reversed(_CREATE_ORDER):
        model.__table__.drop(bind=bind, checkfirst=True)
