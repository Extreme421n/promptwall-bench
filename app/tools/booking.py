"""get_booking_details tool."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import Booking, Customer
from app.tools.base import ResourceNotFoundError, Tool


class GetBookingDetailsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking_reference: Optional[str] = Field(
        default=None, description="6-character record locator (PNR), e.g. 'AB12CD'"
    )
    customer_id: Optional[int] = Field(
        default=None, description="Internal customer id; returns all bookings for them"
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetBookingDetailsInput":
        provided = sum(x is not None for x in (self.booking_reference, self.customer_id))
        if provided != 1:
            raise ValueError("provide exactly one of booking_reference, customer_id")
        return self


class BookingDetails(BaseModel):
    booking_reference: str
    customer_id: int
    customer_name: str
    flight_number: str
    booking_status: str
    cabin_class: str
    total_paid: Decimal
    currency: str
    scheduled_departure: datetime
    scheduled_arrival: datetime


class GetBookingDetailsOutput(BaseModel):
    count: int
    bookings: list[BookingDetails]


def _impl(session: Session, inp: GetBookingDetailsInput) -> GetBookingDetailsOutput:
    stmt = (
        select(Booking)
        .options(selectinload(Booking.flight), selectinload(Booking.customer))
    )
    if inp.booking_reference is not None:
        stmt = stmt.where(Booking.booking_reference == inp.booking_reference)
    else:
        # by customer_id: verify customer exists, then list bookings
        exists = session.execute(
            select(Customer.id).where(Customer.id == inp.customer_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("customer not found")
        stmt = stmt.where(Booking.customer_id == inp.customer_id)

    rows = session.execute(stmt).scalars().all()
    if inp.booking_reference is not None and not rows:
        raise ResourceNotFoundError(f"booking {inp.booking_reference!r} not found")

    items = [
        BookingDetails(
            booking_reference=b.booking_reference,
            customer_id=b.customer_id,
            customer_name=b.customer.full_name,
            flight_number=b.flight.flight_number,
            booking_status=b.booking_status,
            cabin_class=b.cabin_class,
            total_paid=b.total_paid,
            currency=b.currency,
            scheduled_departure=b.flight.scheduled_departure,
            scheduled_arrival=b.flight.scheduled_arrival,
        )
        for b in rows
    ]
    return GetBookingDetailsOutput(count=len(items), bookings=items)


get_booking_details = Tool(
    name="get_booking_details",
    description=(
        "Look up booking details by booking_reference (returns a single booking) "
        "or by customer_id (returns all of that customer's bookings)."
    ),
    domain="airline",
    input_schema=GetBookingDetailsInput,
    output_schema=GetBookingDetailsOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
