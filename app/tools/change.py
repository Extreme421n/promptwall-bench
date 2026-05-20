"""calculate_change_fee and search_change_options tools."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, aliased, selectinload

from app.models import Airport, Booking, Flight, Seat
from app.tools.base import ResourceNotFoundError, Tool

# ---------------------------------------------------------------------------
# calculate_change_fee
# ---------------------------------------------------------------------------

_CHANGE_FEE_BY_CABIN: dict[str, Decimal] = {
    "economy": Decimal("200.00"),
    "premium_economy": Decimal("100.00"),
    "business": Decimal("50.00"),
    "first": Decimal("0.00"),
}

_CABIN_PRICE_BASE: dict[str, Decimal] = {
    "economy": Decimal("200.00"),
    "premium_economy": Decimal("500.00"),
    "business": Decimal("1500.00"),
    "first": Decimal("4500.00"),
}


class CalculateChangeFeeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking_reference: str = Field(description="6-character record locator (PNR)")
    new_date: Optional[date] = Field(
        default=None,
        description="Optional new departure date (YYYY-MM-DD).",
    )
    new_flight_number: Optional[str] = Field(
        default=None,
        description="Optional explicit new flight number for fare-diff estimation.",
    )


class CalculateChangeFeeOutput(BaseModel):
    booking_reference: str
    cabin_class: str
    change_fee: Decimal
    new_fare_difference: Optional[Decimal]
    total_change_cost: Decimal
    currency: str
    notes: str


def _calculate_change_fee_impl(
    session: Session, inp: CalculateChangeFeeInput
) -> CalculateChangeFeeOutput:
    booking = session.execute(
        select(Booking)
        .options(selectinload(Booking.flight))
        .where(Booking.booking_reference == inp.booking_reference)
    ).scalar_one_or_none()
    if booking is None:
        raise ResourceNotFoundError(f"booking {inp.booking_reference!r} not found")

    cabin = booking.cabin_class
    base_fee = _CHANGE_FEE_BY_CABIN.get(cabin, Decimal("250.00"))
    notes_parts: list[str] = [
        f"Change fee derived from cabin class {cabin!r}."
    ]

    new_fare_difference: Optional[Decimal] = None
    if inp.new_flight_number is not None:
        new_flight = session.execute(
            select(Flight)
            .where(Flight.flight_number == inp.new_flight_number)
            .order_by(Flight.scheduled_departure)
            .limit(1)
        ).scalar_one_or_none()
        if new_flight is None:
            notes_parts.append(
                f"New flight {inp.new_flight_number!r} not found; "
                "fare difference omitted."
            )
        else:
            estimated_new_fare = _CABIN_PRICE_BASE.get(cabin, Decimal("250.00"))
            diff = (estimated_new_fare - booking.total_paid).quantize(Decimal("0.01"))
            # Only positive differences are charged; negatives are absorbed.
            new_fare_difference = diff if diff > 0 else Decimal("0.00")
            notes_parts.append(
                f"Estimated fare diff vs current total paid: {diff} {booking.currency}."
            )
    elif inp.new_date is not None:
        notes_parts.append(
            f"Date-only change requested for {inp.new_date.isoformat()}; "
            "fare difference cannot be estimated without a new flight selection."
        )

    total = base_fee + (new_fare_difference or Decimal("0.00"))
    return CalculateChangeFeeOutput(
        booking_reference=booking.booking_reference,
        cabin_class=cabin,
        change_fee=base_fee,
        new_fare_difference=new_fare_difference,
        total_change_cost=total.quantize(Decimal("0.01")),
        currency=booking.currency,
        notes=" ".join(notes_parts),
    )


calculate_change_fee = Tool(
    name="calculate_change_fee",
    description=(
        "Estimate the change fee for a booking. Optionally accepts a new_date "
        "or new_flight_number to refine the estimate. Returns the change fee, "
        "estimated fare difference (if a new flight is specified), and total."
    ),
    domain="airline",
    input_schema=CalculateChangeFeeInput,
    output_schema=CalculateChangeFeeOutput,
    risk_level="medium",
    read_only=True,
    impl=_calculate_change_fee_impl,
)


# ---------------------------------------------------------------------------
# search_change_options
# ---------------------------------------------------------------------------


class SearchChangeOptionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking_reference: str = Field(description="6-character record locator (PNR)")
    date_from: date = Field(description="Inclusive earliest new-departure date.")
    date_to: date = Field(description="Inclusive latest new-departure date.")

    @model_validator(mode="after")
    def _validate(self) -> "SearchChangeOptionsInput":
        if self.date_to < self.date_from:
            raise ValueError("date_to must be on or after date_from")
        return self


class ChangeOptionItem(BaseModel):
    flight_number: str
    origin_code: str
    destination_code: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    status: str
    available_seats_in_cabin: int


class SearchChangeOptionsOutput(BaseModel):
    booking_reference: str
    origin_code: str
    destination_code: str
    cabin_class: str
    count: int
    options: list[ChangeOptionItem]


def _search_change_options_impl(
    session: Session, inp: SearchChangeOptionsInput
) -> SearchChangeOptionsOutput:
    booking = session.execute(
        select(Booking)
        .options(selectinload(Booking.flight))
        .where(Booking.booking_reference == inp.booking_reference)
    ).scalar_one_or_none()
    if booking is None:
        raise ResourceNotFoundError(f"booking {inp.booking_reference!r} not found")

    current_flight = booking.flight
    origin = aliased(Airport)
    destination = aliased(Airport)

    start = datetime.combine(inp.date_from, time.min, tzinfo=timezone.utc)
    end = datetime.combine(inp.date_to, time.max, tzinfo=timezone.utc)

    rows = session.execute(
        select(Flight, origin.code, destination.code)
        .join(origin, Flight.origin_airport_id == origin.id)
        .join(destination, Flight.destination_airport_id == destination.id)
        .where(
            and_(
                Flight.origin_airport_id == current_flight.origin_airport_id,
                Flight.destination_airport_id == current_flight.destination_airport_id,
                Flight.id != current_flight.id,
                Flight.scheduled_departure >= start,
                Flight.scheduled_departure <= end,
                Flight.status.in_(("scheduled", "delayed", "boarding")),
            )
        )
        .order_by(Flight.scheduled_departure)
        .limit(25)
    ).all()

    origin_code = session.execute(
        select(Airport.code).where(Airport.id == current_flight.origin_airport_id)
    ).scalar_one()
    destination_code = session.execute(
        select(Airport.code).where(Airport.id == current_flight.destination_airport_id)
    ).scalar_one()

    if not rows:
        return SearchChangeOptionsOutput(
            booking_reference=booking.booking_reference,
            origin_code=origin_code,
            destination_code=destination_code,
            cabin_class=booking.cabin_class,
            count=0,
            options=[],
        )

    flight_ids = [f.id for f, _, _ in rows]
    seat_counts: dict[int, int] = dict(
        session.execute(
            select(Seat.flight_id, func.count())
            .where(
                Seat.flight_id.in_(flight_ids),
                Seat.is_available.is_(True),
                Seat.cabin_class == booking.cabin_class,
            )
            .group_by(Seat.flight_id)
        ).all()
    )

    options = [
        ChangeOptionItem(
            flight_number=f.flight_number,
            origin_code=oc,
            destination_code=dc,
            scheduled_departure=f.scheduled_departure,
            scheduled_arrival=f.scheduled_arrival,
            status=f.status,
            available_seats_in_cabin=seat_counts.get(f.id, 0),
        )
        for f, oc, dc in rows
    ]

    return SearchChangeOptionsOutput(
        booking_reference=booking.booking_reference,
        origin_code=origin_code,
        destination_code=destination_code,
        cabin_class=booking.cabin_class,
        count=len(options),
        options=options,
    )


search_change_options = Tool(
    name="search_change_options",
    description=(
        "Find alternative flights for an existing booking on the same route, "
        "within a date range. Returns up to 25 candidate flights with the "
        "count of available seats in the booking's cabin class."
    ),
    domain="airline",
    input_schema=SearchChangeOptionsInput,
    output_schema=SearchChangeOptionsOutput,
    risk_level="medium",
    read_only=True,
    impl=_search_change_options_impl,
)
