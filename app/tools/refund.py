"""get_refund_status tool."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking, Customer, Refund
from app.tools.base import ResourceNotFoundError, Tool


class GetRefundStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    booking_reference: Optional[str] = Field(default=None)
    customer_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetRefundStatusInput":
        provided = sum(x is not None for x in (self.booking_reference, self.customer_id))
        if provided != 1:
            raise ValueError("provide exactly one of booking_reference, customer_id")
        return self


class RefundItem(BaseModel):
    booking_reference: str
    refund_status: str
    refund_amount: Decimal
    currency: str
    reason: Optional[str]
    expected_resolution_date: Optional[date]


class GetRefundStatusOutput(BaseModel):
    count: int
    refunds: list[RefundItem]


def _impl(session: Session, inp: GetRefundStatusInput) -> GetRefundStatusOutput:
    stmt = (
        select(Refund, Booking.booking_reference, Booking.currency)
        .join(Booking, Refund.booking_id == Booking.id)
    )
    if inp.booking_reference is not None:
        booking = session.execute(
            select(Booking.id).where(Booking.booking_reference == inp.booking_reference)
        ).scalar_one_or_none()
        if booking is None:
            raise ResourceNotFoundError(f"booking {inp.booking_reference!r} not found")
        stmt = stmt.where(Refund.booking_id == booking)
    else:
        exists = session.execute(
            select(Customer.id).where(Customer.id == inp.customer_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("customer not found")
        stmt = stmt.where(Booking.customer_id == inp.customer_id)

    rows = session.execute(stmt).all()
    items = [
        RefundItem(
            booking_reference=ref,
            refund_status=r.refund_status,
            refund_amount=r.refund_amount,
            currency=cur,
            reason=r.reason,
            expected_resolution_date=r.expected_resolution_date,
        )
        for r, ref, cur in rows
    ]
    return GetRefundStatusOutput(count=len(items), refunds=items)


get_refund_status = Tool(
    name="get_refund_status",
    description=(
        "Look up refund records by booking_reference (single booking) or "
        "customer_id (all of that customer's refunds). Returns status, amount, "
        "expected resolution date, and reason."
    ),
    domain="airline",
    input_schema=GetRefundStatusInput,
    output_schema=GetRefundStatusOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
