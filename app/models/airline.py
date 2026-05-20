"""Airline domain tables: airports, flights, bookings, seats, baggage, refunds."""

from __future__ import annotations

from datetime import datetime
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


class Airport(Base):
    __tablename__ = "airports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(8), nullable=False, unique=True)
    city: Mapped[str] = mapped_column(String(80), nullable=False)
    country: Mapped[str] = mapped_column(String(80), nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False)


class Flight(Base):
    __tablename__ = "flights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flight_number: Mapped[str] = mapped_column(String(10), nullable=False)
    origin_airport_id: Mapped[int] = mapped_column(
        ForeignKey("airports.id", ondelete="RESTRICT"), nullable=False
    )
    destination_airport_id: Mapped[int] = mapped_column(
        ForeignKey("airports.id", ondelete="RESTRICT"), nullable=False
    )
    scheduled_departure: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    scheduled_arrival: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="scheduled")
    gate: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    origin_airport: Mapped["Airport"] = relationship(foreign_keys=[origin_airport_id])
    destination_airport: Mapped["Airport"] = relationship(foreign_keys=[destination_airport_id])
    bookings: Mapped[list["Booking"]] = relationship(back_populates="flight")
    seats: Mapped[list["Seat"]] = relationship(
        back_populates="flight", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_flights_flight_number", "flight_number"),
        Index("ix_flights_scheduled_departure", "scheduled_departure"),
    )


class Booking(Base, TimestampMixin):
    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_reference: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False
    )
    flight_id: Mapped[int] = mapped_column(
        ForeignKey("flights.id", ondelete="RESTRICT"), nullable=False
    )
    booking_status: Mapped[str] = mapped_column(String(20), nullable=False, default="confirmed")
    cabin_class: Mapped[str] = mapped_column(String(20), nullable=False, default="economy")
    total_paid: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")

    customer: Mapped["Customer"] = relationship(back_populates="bookings")
    flight: Mapped["Flight"] = relationship(back_populates="bookings")
    refunds: Mapped[list["Refund"]] = relationship(
        back_populates="booking", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_bookings_customer_id", "customer_id"),
        Index("ix_bookings_flight_id", "flight_id"),
    )


class Seat(Base):
    __tablename__ = "seats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flight_id: Mapped[int] = mapped_column(
        ForeignKey("flights.id", ondelete="CASCADE"), nullable=False
    )
    seat_number: Mapped[str] = mapped_column(String(8), nullable=False)
    cabin_class: Mapped[str] = mapped_column(String(20), nullable=False, default="economy")
    is_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    flight: Mapped["Flight"] = relationship(back_populates="seats")

    __table_args__ = (
        UniqueConstraint("flight_id", "seat_number", name="uq_seats_flight_seat"),
        Index("ix_seats_flight_id", "flight_id"),
    )


class BaggageRule(Base):
    __tablename__ = "baggage_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    route_type: Mapped[str] = mapped_column(String(20), nullable=False)  # domestic / international
    cabin_class: Mapped[str] = mapped_column(String(20), nullable=False)
    checked_bag_kg: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cabin_bag_kg: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    policy_text: Mapped[str] = mapped_column(Text, nullable=False)
    effective_from: Mapped[datetime] = mapped_column(Date, nullable=False)

    __table_args__ = (
        Index("ix_baggage_rules_route_cabin", "route_type", "cabin_class"),
    )


class Refund(Base):
    __tablename__ = "refunds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False
    )
    refund_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    refund_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    expected_resolution_date: Mapped[Optional[datetime]] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    booking: Mapped["Booking"] = relationship(back_populates="refunds")

    __table_args__ = (
        Index("ix_refunds_booking_id", "booking_id"),
    )
