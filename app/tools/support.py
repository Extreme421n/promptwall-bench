"""get_support_ticket_status tool."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.models import Customer, SupportMessage, SupportTicket
from app.tools.base import ResourceNotFoundError, Tool


class GetSupportTicketStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_number: Optional[str] = Field(
        default=None, description="e.g. 'TKT-AB12CD'"
    )
    customer_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetSupportTicketStatusInput":
        provided = sum(x is not None for x in (self.ticket_number, self.customer_id))
        if provided != 1:
            raise ValueError("provide exactly one of ticket_number, customer_id")
        return self


class SupportTicketItem(BaseModel):
    ticket_number: str
    customer_id: int
    subject: str
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime
    last_message_excerpt: Optional[str]
    last_message_sender: Optional[str]
    message_count: int


class GetSupportTicketStatusOutput(BaseModel):
    count: int
    tickets: list[SupportTicketItem]


def _excerpt(text: str, max_len: int = 200) -> str:
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _impl(
    session: Session, inp: GetSupportTicketStatusInput
) -> GetSupportTicketStatusOutput:
    stmt = select(SupportTicket)
    if inp.ticket_number is not None:
        stmt = stmt.where(SupportTicket.ticket_number == inp.ticket_number)
    else:
        exists = session.execute(
            select(Customer.id).where(Customer.id == inp.customer_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("customer not found")
        stmt = stmt.where(SupportTicket.customer_id == inp.customer_id)
    stmt = stmt.order_by(SupportTicket.updated_at.desc())

    tickets = session.execute(stmt).scalars().all()
    if inp.ticket_number is not None and not tickets:
        raise ResourceNotFoundError(f"ticket {inp.ticket_number!r} not found")

    items: list[SupportTicketItem] = []
    for t in tickets:
        last_msg = session.execute(
            select(SupportMessage)
            .where(SupportMessage.ticket_id == t.id)
            .order_by(desc(SupportMessage.created_at))
            .limit(1)
        ).scalar_one_or_none()
        items.append(
            SupportTicketItem(
                ticket_number=t.ticket_number,
                customer_id=t.customer_id,
                subject=t.subject,
                status=t.status,
                priority=t.priority,
                created_at=t.created_at,
                updated_at=t.updated_at,
                last_message_excerpt=_excerpt(last_msg.body) if last_msg else None,
                last_message_sender=last_msg.sender_type if last_msg else None,
                message_count=len(t.messages),
            )
        )
    return GetSupportTicketStatusOutput(count=len(items), tickets=items)


get_support_ticket_status = Tool(
    name="get_support_ticket_status",
    description=(
        "Look up support ticket(s) by ticket_number (single ticket) or "
        "customer_id (all of that customer's tickets, newest first). Each ticket "
        "includes a short excerpt of the most recent message."
    ),
    domain="support",
    input_schema=GetSupportTicketStatusInput,
    output_schema=GetSupportTicketStatusOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
