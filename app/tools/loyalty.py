"""get_loyalty_balance tool — a deliberate narrow subset of get_customer_profile.

The chatbot has both available, so it has to choose. A correct call to either
tool can answer a 'loyalty balance' question.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Customer, LoyaltyAccount
from app.tools.base import ResourceNotFoundError, Tool


class GetLoyaltyBalanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: Optional[int] = None
    email: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetLoyaltyBalanceInput":
        provided = sum(x is not None for x in (self.customer_id, self.email))
        if provided != 1:
            raise ValueError("provide exactly one of customer_id, email")
        return self


class GetLoyaltyBalanceOutput(BaseModel):
    customer_id: int
    has_loyalty: bool
    loyalty_number: Optional[str] = None
    tier: Optional[str] = None
    points_balance: Optional[int] = None


def _impl(session: Session, inp: GetLoyaltyBalanceInput) -> GetLoyaltyBalanceOutput:
    stmt = select(Customer)
    if inp.customer_id is not None:
        stmt = stmt.where(Customer.id == inp.customer_id)
    else:
        stmt = stmt.where(Customer.email == inp.email)

    customer = session.execute(stmt).scalar_one_or_none()
    if customer is None:
        raise ResourceNotFoundError("customer not found")

    loyalty = session.execute(
        select(LoyaltyAccount).where(LoyaltyAccount.customer_id == customer.id)
    ).scalar_one_or_none()

    if loyalty is None:
        return GetLoyaltyBalanceOutput(customer_id=customer.id, has_loyalty=False)

    return GetLoyaltyBalanceOutput(
        customer_id=customer.id,
        has_loyalty=True,
        loyalty_number=loyalty.loyalty_number,
        tier=loyalty.tier,
        points_balance=loyalty.points_balance,
    )


get_loyalty_balance = Tool(
    name="get_loyalty_balance",
    description=(
        "Look up a customer's loyalty tier and points balance by customer_id "
        "or email. Returns has_loyalty=false when the customer has no loyalty "
        "account."
    ),
    domain="crm",
    input_schema=GetLoyaltyBalanceInput,
    output_schema=GetLoyaltyBalanceOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
