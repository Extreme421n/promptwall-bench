"""Textual knowledge schema (Phase 6B-1).

Adds enterprise text artifacts the chatbot will eventually need to ground
answers against: policy documents and their clauses, product warranty terms,
return rules, internal agent notes, operational incidents, and support
resolution templates.

Phase 6B-1 is schema + tests only. No tools, no chatbot changes, no eval
cases, no PromptWall logic.
"""

from __future__ import annotations

from datetime import date as _date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.commerce import Product, ProductCategory
    from app.models.crm import Customer


# String enums (kept loose so future additions don't need migrations).
VALID_KNOWLEDGE_DOMAINS = (
    "airline",
    "commerce",
    "saas",
    "support",
    "crm",
)

VALID_POLICY_TYPES = (
    "refund_policy",
    "return_policy",
    "cancellation_policy",
    "baggage_policy",
    "privacy_policy",
    "overage_policy",
    "warranty_policy",
    "escalation_policy",
    "subscription_policy",
    "payment_policy",
)


# ---------------------------------------------------------------------------
# Policy documents + clauses
# ---------------------------------------------------------------------------


class PolicyDocument(Base, TimestampMixin):
    __tablename__ = "policy_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain: Mapped[str] = mapped_column(String(40), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    policy_type: Mapped[str] = mapped_column(String(40), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    effective_from: Mapped[_date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[Optional[_date]] = mapped_column(Date, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    clauses: Mapped[list["PolicyClause"]] = relationship(
        back_populates="policy_document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_policy_documents_domain_type_active",
            "domain",
            "policy_type",
            "is_active",
        ),
        Index("ix_policy_documents_domain", "domain"),
        Index("ix_policy_documents_policy_type", "policy_type"),
    )


class PolicyClause(Base, TimestampMixin):
    __tablename__ = "policy_clauses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    policy_document_id: Mapped[int] = mapped_column(
        ForeignKey("policy_documents.id", ondelete="CASCADE"), nullable=False
    )
    clause_key: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="normal")
    applies_to: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    exceptions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    policy_document: Mapped["PolicyDocument"] = relationship(back_populates="clauses")

    __table_args__ = (
        Index("ix_policy_clauses_policy_document_id", "policy_document_id"),
        Index("ix_policy_clauses_clause_key", "clause_key"),
    )


# ---------------------------------------------------------------------------
# Commerce text artifacts
# ---------------------------------------------------------------------------


class ProductWarrantyTerms(Base, TimestampMixin):
    __tablename__ = "product_warranty_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warranty_type: Mapped[str] = mapped_column(String(40), nullable=False)
    duration_months: Mapped[int] = mapped_column(Integer, nullable=False, default=12)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    exclusions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_product_warranty_terms_product_id", "product_id"),
    )


class ProductReturnRule(Base, TimestampMixin):
    __tablename__ = "product_return_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_category_id: Mapped[int] = mapped_column(
        ForeignKey("product_categories.id", ondelete="CASCADE"), nullable=False
    )
    rule_name: Mapped[str] = mapped_column(String(120), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    opened_item_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    return_window_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    restocking_fee_percent: Mapped[Decimal] = mapped_column(
        Numeric(5, 2), nullable=False, default=0
    )
    exceptions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_product_return_rules_product_category_id", "product_category_id"),
    )


# ---------------------------------------------------------------------------
# Customer-scoped + operational text
# ---------------------------------------------------------------------------


class InternalAgentNote(Base, TimestampMixin):
    """Free-text notes that human/AI agents leave on a customer record.

    ``related_type`` + ``related_id`` form a polymorphic pointer to the
    object the note was made *about* (e.g. ``booking`` / ``order`` /
    ``invoice`` / ``ticket``). Kept loose because FK constraints across many
    target tables would be over-engineering for a notes table.
    """

    __tablename__ = "internal_agent_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    related_type: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    related_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    note_type: Mapped[str] = mapped_column(String(40), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_internal_agent_notes_customer_id", "customer_id"),
        Index("ix_internal_agent_notes_related", "related_type", "related_id"),
    )


class OperationalIncident(Base, TimestampMixin):
    __tablename__ = "operational_incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain: Mapped[str] = mapped_column(String(40), nullable=False)
    incident_type: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    affected_entities_json: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    __table_args__ = (
        Index("ix_operational_incidents_domain_type", "domain", "incident_type"),
    )


class SupportResolutionTemplate(Base, TimestampMixin):
    __tablename__ = "support_resolution_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    escalation_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_support_resolution_templates_category", "category"),
    )
