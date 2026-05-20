"""get_customer_open_issues tool — open support tickets + pending refunds for one customer."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking, Customer, Refund, SupportTicket
from app.tools.base import ResourceNotFoundError, Tool


_OPEN_TICKET_STATUSES = ("open", "pending")
_PENDING_REFUND_STATUSES = ("pending", "approved")


class GetCustomerOpenIssuesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int = Field(description="Internal numeric customer id")


class OpenIssueItem(BaseModel):
    type: str  # "ticket" | "refund"
    identifier: str
    status: str
    priority: Optional[str] = None
    subject: Optional[str] = None
    refund_amount: Optional[Decimal] = None
    currency: Optional[str] = None


class GetCustomerOpenIssuesOutput(BaseModel):
    customer_id: int
    open_ticket_count: int
    pending_refund_count: int
    issues: list[OpenIssueItem]


def _impl(
    session: Session, inp: GetCustomerOpenIssuesInput
) -> GetCustomerOpenIssuesOutput:
    exists = session.execute(
        select(Customer.id).where(Customer.id == inp.customer_id)
    ).scalar_one_or_none()
    if exists is None:
        raise ResourceNotFoundError("customer not found")

    tickets = session.execute(
        select(SupportTicket)
        .where(
            SupportTicket.customer_id == inp.customer_id,
            SupportTicket.status.in_(_OPEN_TICKET_STATUSES),
        )
        .order_by(SupportTicket.updated_at.desc())
        .limit(25)
    ).scalars().all()

    refunds = session.execute(
        select(Refund, Booking.booking_reference, Booking.currency)
        .join(Booking, Refund.booking_id == Booking.id)
        .where(
            Booking.customer_id == inp.customer_id,
            Refund.refund_status.in_(_PENDING_REFUND_STATUSES),
        )
        .order_by(Refund.updated_at.desc())
        .limit(25)
    ).all()

    issues: list[OpenIssueItem] = []
    for t in tickets:
        issues.append(
            OpenIssueItem(
                type="ticket",
                identifier=t.ticket_number,
                status=t.status,
                priority=t.priority,
                subject=t.subject,
            )
        )
    for r, booking_ref, currency in refunds:
        issues.append(
            OpenIssueItem(
                type="refund",
                identifier=booking_ref,
                status=r.refund_status,
                refund_amount=r.refund_amount,
                currency=currency,
            )
        )

    return GetCustomerOpenIssuesOutput(
        customer_id=inp.customer_id,
        open_ticket_count=len(tickets),
        pending_refund_count=len(refunds),
        issues=issues,
    )


get_customer_open_issues = Tool(
    name="get_customer_open_issues",
    description=(
        "Return a customer's currently-open support tickets and pending "
        "refunds in a single shot. Useful when the user asks 'what's open on "
        "my account?' or an agent needs a quick triage view."
    ),
    domain="support",
    input_schema=GetCustomerOpenIssuesInput,
    output_schema=GetCustomerOpenIssuesOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
