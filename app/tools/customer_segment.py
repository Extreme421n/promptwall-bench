"""get_customer_segment tool (Phase C2).

Returns the customer's declared segment plus a small dashboard of activity
counts across all three business domains (airline bookings, commerce orders,
SaaS organisations). Distinct from get_customer_profile, which is the
identity/loyalty snapshot.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Booking,
    CommerceOrder,
    Customer,
    CustomerOrganization,
    LoyaltyAccount,
    SupportTicket,
)
from app.tools.base import ResourceNotFoundError, Tool


class GetCustomerSegmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: Optional[int] = None
    external_customer_id: Optional[str] = None
    email: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetCustomerSegmentInput":
        provided = sum(
            x is not None for x in (self.customer_id, self.external_customer_id, self.email)
        )
        if provided != 1:
            raise ValueError(
                "provide exactly one of customer_id, external_customer_id, email"
            )
        return self


class GetCustomerSegmentOutput(BaseModel):
    customer_id: int
    external_customer_id: str
    full_name: str
    segment: Optional[str]
    has_loyalty: bool
    loyalty_tier: Optional[str]
    loyalty_points: Optional[int]
    booking_count: int
    commerce_order_count: int
    organization_count: int
    support_ticket_count: int


def _impl(session: Session, inp: GetCustomerSegmentInput) -> GetCustomerSegmentOutput:
    stmt = select(Customer)
    if inp.customer_id is not None:
        stmt = stmt.where(Customer.id == inp.customer_id)
    elif inp.external_customer_id is not None:
        stmt = stmt.where(Customer.external_customer_id == inp.external_customer_id)
    else:
        stmt = stmt.where(Customer.email == inp.email)

    customer = session.execute(stmt).scalar_one_or_none()
    if customer is None:
        raise ResourceNotFoundError("customer not found")

    loyalty = session.execute(
        select(LoyaltyAccount).where(LoyaltyAccount.customer_id == customer.id)
    ).scalar_one_or_none()

    booking_count = int(
        session.execute(
            select(func.count()).select_from(Booking).where(Booking.customer_id == customer.id)
        ).scalar_one()
    )
    order_count = int(
        session.execute(
            select(func.count())
            .select_from(CommerceOrder)
            .where(CommerceOrder.customer_id == customer.id)
        ).scalar_one()
    )
    org_count = int(
        session.execute(
            select(func.count())
            .select_from(CustomerOrganization)
            .where(CustomerOrganization.customer_id == customer.id)
        ).scalar_one()
    )
    ticket_count = int(
        session.execute(
            select(func.count())
            .select_from(SupportTicket)
            .where(SupportTicket.customer_id == customer.id)
        ).scalar_one()
    )

    return GetCustomerSegmentOutput(
        customer_id=customer.id,
        external_customer_id=customer.external_customer_id,
        full_name=customer.full_name,
        segment=customer.segment,
        has_loyalty=loyalty is not None,
        loyalty_tier=loyalty.tier if loyalty else None,
        loyalty_points=loyalty.points_balance if loyalty else None,
        booking_count=booking_count,
        commerce_order_count=order_count,
        organization_count=org_count,
        support_ticket_count=ticket_count,
    )


get_customer_segment = Tool(
    name="get_customer_segment",
    description=(
        "Return the customer's declared segment plus per-domain activity "
        "counts (airline bookings, commerce orders, SaaS organisation "
        "memberships, support tickets). Useful for triage. Lookup by "
        "customer_id, external_customer_id, or email."
    ),
    domain="crm",
    input_schema=GetCustomerSegmentInput,
    output_schema=GetCustomerSegmentOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
