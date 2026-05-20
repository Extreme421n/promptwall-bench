"""get_customer_profile tool."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Customer
from app.tools.base import ResourceNotFoundError, Tool


class GetCustomerProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: Optional[int] = Field(default=None, description="Internal numeric customer id")
    external_customer_id: Optional[str] = Field(
        default=None, description="External CRM id, e.g. CUST-00001"
    )
    email: Optional[str] = Field(default=None, description="Customer email")

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetCustomerProfileInput":
        provided = sum(
            x is not None for x in (self.customer_id, self.external_customer_id, self.email)
        )
        if provided != 1:
            raise ValueError(
                "provide exactly one of customer_id, external_customer_id, email"
            )
        return self


class GetCustomerProfileOutput(BaseModel):
    customer_id: int
    external_customer_id: str
    full_name: str
    email: str
    phone: Optional[str]
    segment: Optional[str]
    has_loyalty: bool
    loyalty_tier: Optional[str] = None
    loyalty_points: Optional[int] = None
    loyalty_number: Optional[str] = None


def _impl(session: Session, inp: GetCustomerProfileInput) -> GetCustomerProfileOutput:
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

    loyalty = customer.loyalty_account
    return GetCustomerProfileOutput(
        customer_id=customer.id,
        external_customer_id=customer.external_customer_id,
        full_name=customer.full_name,
        email=customer.email,
        phone=customer.phone,
        segment=customer.segment,
        has_loyalty=loyalty is not None,
        loyalty_tier=loyalty.tier if loyalty else None,
        loyalty_points=loyalty.points_balance if loyalty else None,
        loyalty_number=loyalty.loyalty_number if loyalty else None,
    )


get_customer_profile = Tool(
    name="get_customer_profile",
    description=(
        "Look up a customer's profile by internal customer_id, external_customer_id "
        "(e.g. 'CUST-00001'), or email. Returns identity, segment, and loyalty status."
    ),
    domain="crm",
    input_schema=GetCustomerProfileInput,
    output_schema=GetCustomerProfileOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
