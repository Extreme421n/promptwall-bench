"""get_saas_seat_allocation tool (Phase C1).

Deliberately distinct from ``search_available_seats`` (airline). The chatbot
must pick the right tool based on whether ``seats`` means SaaS user seats
or airline seats.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Organization, SeatAllocation
from app.tools.base import ResourceNotFoundError, Tool


class GetSaasSeatAllocationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: int = Field(description="Internal organization id")


class GetSaasSeatAllocationOutput(BaseModel):
    organization_id: int
    organization_name: str
    allocated_seats: int
    used_seats: int
    remaining_seats: int
    updated_at: datetime


def _impl(
    session: Session, inp: GetSaasSeatAllocationInput
) -> GetSaasSeatAllocationOutput:
    org = session.execute(
        select(Organization).where(Organization.id == inp.organization_id)
    ).scalar_one_or_none()
    if org is None:
        raise ResourceNotFoundError("organization not found")
    alloc = session.execute(
        select(SeatAllocation).where(
            SeatAllocation.organization_id == inp.organization_id
        )
    ).scalar_one_or_none()
    if alloc is None:
        raise ResourceNotFoundError("organization has no seat allocation on file")

    remaining = max(0, alloc.allocated_seats - alloc.used_seats)
    return GetSaasSeatAllocationOutput(
        organization_id=org.id,
        organization_name=org.name,
        allocated_seats=alloc.allocated_seats,
        used_seats=alloc.used_seats,
        remaining_seats=remaining,
        updated_at=alloc.updated_at,
    )


get_saas_seat_allocation = Tool(
    name="get_saas_seat_allocation",
    description=(
        "Return the SaaS user-seat allocation for an organization "
        "(allocated, used, remaining). NOT an airline seat lookup — for "
        "flight seats use search_available_seats."
    ),
    domain="saas",
    input_schema=GetSaasSeatAllocationInput,
    output_schema=GetSaasSeatAllocationOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
