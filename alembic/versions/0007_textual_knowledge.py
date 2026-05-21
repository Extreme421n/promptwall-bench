"""add textual knowledge schema

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-20

Phase 6B-1 — adds 7 text-heavy tables for enterprise knowledge: policy
documents + clauses, product warranty terms + return rules, internal agent
notes, operational incidents, support resolution templates.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from app.models import (
    InternalAgentNote,
    OperationalIncident,
    PolicyClause,
    PolicyDocument,
    ProductReturnRule,
    ProductWarrantyTerms,
    SupportResolutionTemplate,
)

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Creation order honours FK dependencies (policy_clauses → policy_documents,
# product_warranty_terms → products, product_return_rules → product_categories,
# internal_agent_notes → customers). Drop order is the reverse.
_CREATE_ORDER = (
    PolicyDocument,                # parent of PolicyClause
    PolicyClause,                  # FK -> policy_documents
    ProductWarrantyTerms,          # FK -> products
    ProductReturnRule,             # FK -> product_categories
    InternalAgentNote,             # FK -> customers
    OperationalIncident,
    SupportResolutionTemplate,
)


def upgrade() -> None:
    bind = op.get_bind()
    for model in _CREATE_ORDER:
        model.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for model in reversed(_CREATE_ORDER):
        model.__table__.drop(bind=bind, checkfirst=True)
