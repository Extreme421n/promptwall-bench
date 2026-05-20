"""SaaS / billing domain models.

Adds a second business domain alongside the airline + support data so the
chatbot benchmark exercises naturally overlapping vocabulary:

* ``seats`` can mean an airline seat (``seats`` table) **or** a SaaS user
  seat (``seat_allocations``).
* ``plan`` can mean a travel plan or a SaaS subscription plan
  (``plans``).
* ``charge`` can mean a flight change fee, an invoice line, or an
  overage charge (``overage_charges``).
* ``status`` covers subscriptions, invoices, flights, bookings, tickets,
  and refunds.

This phase adds the schema and seed only — no new tools, no chatbot
behaviour changes.
"""

from __future__ import annotations

from datetime import date as _date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.crm import Customer


# ---------------------------------------------------------------------------
# organizations + membership
# ---------------------------------------------------------------------------


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    external_org_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    customer_links: Mapped[list["CustomerOrganization"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    subscriptions: Mapped[list["Subscription"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    invoices: Mapped[list["Invoice"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )
    seat_allocation: Mapped[Optional["SeatAllocation"]] = relationship(
        back_populates="organization", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_organizations_name", "name"),)


class CustomerOrganization(Base, TimestampMixin):
    """Membership of a customer in an organization (many-to-many)."""

    __tablename__ = "customer_organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(40), nullable=False, default="member")

    organization: Mapped["Organization"] = relationship(back_populates="customer_links")

    __table_args__ = (
        UniqueConstraint(
            "customer_id", "organization_id", name="uq_customer_organizations_pair"
        ),
        Index("ix_customer_organizations_customer_id", "customer_id"),
        Index("ix_customer_organizations_organization_id", "organization_id"),
    )


# ---------------------------------------------------------------------------
# plans + subscriptions
# ---------------------------------------------------------------------------


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False, unique=True)
    tier: Mapped[str] = mapped_column(String(40), nullable=False)
    monthly_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, default=0)
    included_seats: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    included_api_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    overage_price_per_1000_calls: Mapped[Decimal] = mapped_column(
        Numeric(10, 4), nullable=False, default=0
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (Index("ix_plans_tier", "tier"),)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("plans.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    renews_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    canceled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    organization: Mapped["Organization"] = relationship(back_populates="subscriptions")
    plan: Mapped["Plan"] = relationship()
    invoices: Mapped[list["Invoice"]] = relationship(back_populates="subscription")

    __table_args__ = (
        Index("ix_subscriptions_organization_id", "organization_id"),
        Index("ix_subscriptions_status", "status"),
    )


# ---------------------------------------------------------------------------
# invoices + items + overages
# ---------------------------------------------------------------------------


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    subscription_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("subscriptions.id", ondelete="SET NULL"), nullable=True
    )
    invoice_number: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="issued")
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    paid_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    organization: Mapped["Organization"] = relationship(back_populates="invoices")
    subscription: Mapped[Optional["Subscription"]] = relationship(back_populates="invoices")
    items: Mapped[list["InvoiceItem"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )
    overage_charges: Mapped[list["OverageCharge"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_invoices_organization_id", "organization_id"),
        Index("ix_invoices_status", "status"),
    )


class InvoiceItem(Base):
    __tablename__ = "invoice_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    invoice: Mapped["Invoice"] = relationship(back_populates="items")

    __table_args__ = (Index("ix_invoice_items_invoice_id", "invoice_id"),)


class OverageCharge(Base, TimestampMixin):
    __tablename__ = "overage_charges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(
        ForeignKey("invoices.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    usage_amount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    charge_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    invoice: Mapped["Invoice"] = relationship(back_populates="overage_charges")

    __table_args__ = (
        Index("ix_overage_charges_invoice_id", "invoice_id"),
        Index("ix_overage_charges_organization_id", "organization_id"),
    )


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_usage_events_organization_id", "organization_id"),
        Index("ix_usage_events_event_type", "event_type"),
        Index("ix_usage_events_occurred_at", "occurred_at"),
    )


class ApiUsageDaily(Base):
    __tablename__ = "api_usage_daily"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[_date] = mapped_column(Date, nullable=False)
    api_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("organization_id", "date", name="uq_api_usage_daily_org_date"),
        Index("ix_api_usage_daily_organization_id", "organization_id"),
    )


class SeatAllocation(Base):
    """How many SaaS user seats an org has allocated vs. used.

    Distinct from ``seats`` (airline seats on a flight) — the deliberate
    vocabulary collision is the point.
    """

    __tablename__ = "seat_allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    allocated_seats: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    used_seats: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    organization: Mapped["Organization"] = relationship(back_populates="seat_allocation")
