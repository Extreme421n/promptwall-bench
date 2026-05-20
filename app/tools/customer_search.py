"""search_customer_records tool — fuzzy customer lookup that may return multiple matches."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Customer
from app.tools.base import Tool


class SearchCustomerRecordsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    full_name: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _at_least_one(self) -> "SearchCustomerRecordsInput":
        if not any((self.email, self.phone, self.full_name)):
            raise ValueError("provide at least one of email, phone, full_name")
        return self


class CustomerRecordItem(BaseModel):
    customer_id: int
    external_customer_id: str
    full_name: str
    email: str
    phone: Optional[str]
    segment: Optional[str]


class SearchCustomerRecordsOutput(BaseModel):
    count: int
    matches: list[CustomerRecordItem]


def _impl(session: Session, inp: SearchCustomerRecordsInput) -> SearchCustomerRecordsOutput:
    filters = []
    if inp.email is not None:
        filters.append(Customer.email.ilike(f"%{inp.email.strip()}%"))
    if inp.phone is not None:
        filters.append(Customer.phone.ilike(f"%{inp.phone.strip()}%"))
    if inp.full_name is not None:
        filters.append(Customer.full_name.ilike(f"%{inp.full_name.strip()}%"))

    stmt = (
        select(Customer)
        .where(or_(*filters))
        .order_by(Customer.id)
        .limit(25)
    )
    rows = session.execute(stmt).scalars().all()
    return SearchCustomerRecordsOutput(
        count=len(rows),
        matches=[
            CustomerRecordItem(
                customer_id=c.id,
                external_customer_id=c.external_customer_id,
                full_name=c.full_name,
                email=c.email,
                phone=c.phone,
                segment=c.segment,
            )
            for c in rows
        ],
    )


search_customer_records = Tool(
    name="search_customer_records",
    description=(
        "Search the customer directory by email, phone, or full name (any "
        "combination). Uses case-insensitive partial match and returns up to "
        "25 matches. Useful when an exact customer id isn't known."
    ),
    domain="crm",
    input_schema=SearchCustomerRecordsInput,
    output_schema=SearchCustomerRecordsOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
