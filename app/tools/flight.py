"""get_flight_status and search_available_flights tools."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, aliased, selectinload

from app.models import Airport, Booking, Flight, Seat
from app.tools.base import ResourceNotFoundError, Tool

# ---------------------------------------------------------------------------
# get_flight_status
# ---------------------------------------------------------------------------


class GetFlightStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flight_number: Optional[str] = Field(
        default=None,
        description="Flight number such as 'BA178'. May match multiple dates.",
    )
    booking_reference: Optional[str] = Field(
        default=None,
        description="Booking reference (PNR). Resolves to exactly one flight.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetFlightStatusInput":
        provided = sum(x is not None for x in (self.flight_number, self.booking_reference))
        if provided != 1:
            raise ValueError("provide exactly one of flight_number, booking_reference")
        return self


class FlightStatusItem(BaseModel):
    flight_number: str
    status: str
    origin_code: str
    destination_code: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    gate: Optional[str]
    updated_at: datetime


class GetFlightStatusOutput(BaseModel):
    count: int
    flights: list[FlightStatusItem]


def _flight_status_impl(session: Session, inp: GetFlightStatusInput) -> GetFlightStatusOutput:
    origin = aliased(Airport)
    destination = aliased(Airport)

    base = (
        select(Flight, origin.code, destination.code)
        .join(origin, Flight.origin_airport_id == origin.id)
        .join(destination, Flight.destination_airport_id == destination.id)
    )

    if inp.booking_reference is not None:
        stmt = base.join(Booking, Booking.flight_id == Flight.id).where(
            Booking.booking_reference == inp.booking_reference
        )
    else:
        stmt = base.where(Flight.flight_number == inp.flight_number).order_by(
            Flight.scheduled_departure
        ).limit(10)

    rows = session.execute(stmt).all()
    if not rows:
        if inp.booking_reference is not None:
            raise ResourceNotFoundError(f"booking {inp.booking_reference!r} not found")
        raise ResourceNotFoundError(f"no flights found for {inp.flight_number!r}")

    items = [
        FlightStatusItem(
            flight_number=f.flight_number,
            status=f.status,
            origin_code=oc,
            destination_code=dc,
            scheduled_departure=f.scheduled_departure,
            scheduled_arrival=f.scheduled_arrival,
            gate=f.gate,
            updated_at=f.updated_at,
        )
        for f, oc, dc in rows
    ]
    return GetFlightStatusOutput(count=len(items), flights=items)


get_flight_status = Tool(
    name="get_flight_status",
    description=(
        "Get the current status of a flight by flight_number (returns up to 10 "
        "upcoming/recent matches sorted by departure) or by booking_reference "
        "(returns the one flight the booking is on)."
    ),
    domain="airline",
    input_schema=GetFlightStatusInput,
    output_schema=GetFlightStatusOutput,
    risk_level="low",
    read_only=True,
    impl=_flight_status_impl,
)


# ---------------------------------------------------------------------------
# search_available_flights
# ---------------------------------------------------------------------------


class SearchAvailableFlightsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: str = Field(description="Origin airport IATA code, e.g. 'JFK'")
    destination: str = Field(description="Destination airport IATA code, e.g. 'LHR'")
    date_from: date = Field(description="Inclusive earliest departure date (YYYY-MM-DD)")
    date_to: date = Field(description="Inclusive latest departure date (YYYY-MM-DD)")
    cabin_class: Optional[str] = Field(
        default=None,
        description="Optional cabin: economy, premium_economy, business, first",
    )

    @model_validator(mode="after")
    def _validate(self) -> "SearchAvailableFlightsInput":
        if self.date_to < self.date_from:
            raise ValueError("date_to must be on or after date_from")
        if self.origin.strip().upper() == self.destination.strip().upper():
            raise ValueError("origin and destination must differ")
        return self


class AvailableFlightItem(BaseModel):
    flight_number: str
    origin_code: str
    destination_code: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    status: str
    cabin_class_filter: Optional[str]
    available_seats: int


class SearchAvailableFlightsOutput(BaseModel):
    count: int
    flights: list[AvailableFlightItem]


def _search_flights_impl(
    session: Session, inp: SearchAvailableFlightsInput
) -> SearchAvailableFlightsOutput:
    origin_code = inp.origin.strip().upper()
    destination_code = inp.destination.strip().upper()

    # Resolve airport codes
    origin_id = session.execute(
        select(Airport.id).where(Airport.code == origin_code)
    ).scalar_one_or_none()
    destination_id = session.execute(
        select(Airport.id).where(Airport.code == destination_code)
    ).scalar_one_or_none()
    if origin_id is None:
        raise ResourceNotFoundError(f"origin airport {origin_code!r} not found")
    if destination_id is None:
        raise ResourceNotFoundError(f"destination airport {destination_code!r} not found")

    start = datetime.combine(inp.date_from, time.min, tzinfo=timezone.utc)
    end = datetime.combine(inp.date_to, time.max, tzinfo=timezone.utc)

    origin_alias = aliased(Airport)
    dest_alias = aliased(Airport)

    flights_stmt = (
        select(Flight, origin_alias.code, dest_alias.code)
        .join(origin_alias, Flight.origin_airport_id == origin_alias.id)
        .join(dest_alias, Flight.destination_airport_id == dest_alias.id)
        .where(
            and_(
                Flight.origin_airport_id == origin_id,
                Flight.destination_airport_id == destination_id,
                Flight.scheduled_departure >= start,
                Flight.scheduled_departure <= end,
                Flight.status.in_(("scheduled", "delayed", "boarding")),
            )
        )
        .order_by(Flight.scheduled_departure)
        .limit(50)
    )
    rows = session.execute(flights_stmt).all()
    if not rows:
        return SearchAvailableFlightsOutput(count=0, flights=[])

    flight_ids = [f.id for f, _, _ in rows]
    seat_filters = [Seat.flight_id.in_(flight_ids), Seat.is_available.is_(True)]
    if inp.cabin_class is not None:
        seat_filters.append(Seat.cabin_class == inp.cabin_class)

    seat_counts: dict[int, int] = dict(
        session.execute(
            select(Seat.flight_id, func.count())
            .where(and_(*seat_filters))
            .group_by(Seat.flight_id)
        ).all()
    )

    items = [
        AvailableFlightItem(
            flight_number=f.flight_number,
            origin_code=oc,
            destination_code=dc,
            scheduled_departure=f.scheduled_departure,
            scheduled_arrival=f.scheduled_arrival,
            status=f.status,
            cabin_class_filter=inp.cabin_class,
            available_seats=seat_counts.get(f.id, 0),
        )
        for f, oc, dc in rows
    ]
    return SearchAvailableFlightsOutput(count=len(items), flights=items)


search_available_flights = Tool(
    name="search_available_flights",
    description=(
        "Find scheduled flights between two airports within a date range. "
        "Returns up to 50 flights with the number of available seats (filtered "
        "by cabin_class if provided)."
    ),
    domain="airline",
    input_schema=SearchAvailableFlightsInput,
    output_schema=SearchAvailableFlightsOutput,
    risk_level="low",
    read_only=True,
    impl=_search_flights_impl,
)
