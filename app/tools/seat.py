"""search_available_seats tool."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking, Flight, Seat
from app.tools.base import ResourceNotFoundError, Tool

_VALID_CABINS = {"economy", "premium_economy", "business", "first"}


class SearchAvailableSeatsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flight_number: Optional[str] = Field(
        default=None,
        description="Flight number; resolves to the next upcoming flight if multiple match.",
    )
    booking_reference: Optional[str] = Field(
        default=None,
        description="Booking reference (PNR); resolves to that booking's flight.",
    )
    cabin_class: Optional[str] = Field(
        default=None,
        description="Optional cabin filter: economy, premium_economy, business, first",
    )

    @model_validator(mode="after")
    def _validate(self) -> "SearchAvailableSeatsInput":
        provided = sum(x is not None for x in (self.flight_number, self.booking_reference))
        if provided != 1:
            raise ValueError("provide exactly one of flight_number, booking_reference")
        if self.cabin_class is not None and self.cabin_class not in _VALID_CABINS:
            raise ValueError(
                f"cabin_class must be one of {sorted(_VALID_CABINS)}"
            )
        return self


class SeatItem(BaseModel):
    seat_number: str
    cabin_class: str


class SearchAvailableSeatsOutput(BaseModel):
    flight_id: int
    flight_number: str
    cabin_filter: Optional[str]
    count: int
    seats: list[SeatItem]


def _impl(session: Session, inp: SearchAvailableSeatsInput) -> SearchAvailableSeatsOutput:
    if inp.booking_reference is not None:
        flight = session.execute(
            select(Flight)
            .join(Booking, Booking.flight_id == Flight.id)
            .where(Booking.booking_reference == inp.booking_reference)
            .limit(1)
        ).scalar_one_or_none()
        if flight is None:
            raise ResourceNotFoundError(
                f"booking {inp.booking_reference!r} not found"
            )
    else:
        # Pick the earliest matching flight (chatbot can refine by date later).
        flight = session.execute(
            select(Flight)
            .where(Flight.flight_number == inp.flight_number)
            .order_by(Flight.scheduled_departure)
            .limit(1)
        ).scalar_one_or_none()
        if flight is None:
            raise ResourceNotFoundError(
                f"no flights found for {inp.flight_number!r}"
            )

    seats_stmt = (
        select(Seat)
        .where(Seat.flight_id == flight.id, Seat.is_available.is_(True))
        .order_by(Seat.seat_number)
    )
    if inp.cabin_class is not None:
        seats_stmt = seats_stmt.where(Seat.cabin_class == inp.cabin_class)
    seats = session.execute(seats_stmt.limit(50)).scalars().all()

    return SearchAvailableSeatsOutput(
        flight_id=flight.id,
        flight_number=flight.flight_number,
        cabin_filter=inp.cabin_class,
        count=len(seats),
        seats=[SeatItem(seat_number=s.seat_number, cabin_class=s.cabin_class) for s in seats],
    )


search_available_seats = Tool(
    name="search_available_seats",
    description=(
        "List available seats on a flight, looked up by flight_number (returns "
        "the earliest matching flight) or by booking_reference. Optionally "
        "filter by cabin_class."
    ),
    domain="airline",
    input_schema=SearchAvailableSeatsInput,
    output_schema=SearchAvailableSeatsOutput,
    risk_level="low",
    read_only=True,
    impl=_impl,
)
