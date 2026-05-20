"""Customer relationship management tables."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.airline import Booking
    from app.models.observability import Trace
    from app.models.support import SupportTicket


class Customer(Base, TimestampMixin):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_customer_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    segment: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)

    loyalty_account: Mapped[Optional["LoyaltyAccount"]] = relationship(
        back_populates="customer",
        uselist=False,
        cascade="all, delete-orphan",
    )
    bookings: Mapped[list["Booking"]] = relationship(back_populates="customer")
    support_tickets: Mapped[list["SupportTicket"]] = relationship(back_populates="customer")
    traces: Mapped[list["Trace"]] = relationship(back_populates="customer")

    __table_args__ = (
        Index("ix_customers_email", "email"),
    )


class LoyaltyAccount(Base):
    __tablename__ = "loyalty_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    loyalty_number: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    tier: Mapped[str] = mapped_column(String(20), nullable=False, default="standard")
    points_balance: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    customer: Mapped["Customer"] = relationship(back_populates="loyalty_account")
