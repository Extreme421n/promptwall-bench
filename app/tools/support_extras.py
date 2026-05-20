"""Phase C2 support tools.

* search_support_tickets       — read-only search by subject/status/customer
* get_escalation_policy        — read-only; static policy table + optional
                                 ticket-priority lookup
* create_support_ticket_draft  — DRAFT ONLY; validates the input and returns a
                                 structured draft. Does not write to the DB.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Customer, SupportTicket
from app.tools.base import ResourceNotFoundError, Tool

_VALID_TICKET_STATUSES = {"open", "pending", "resolved", "closed"}
_VALID_PRIORITIES = {"low", "normal", "high", "urgent"}


# ---------------------------------------------------------------------------
# search_support_tickets
# ---------------------------------------------------------------------------


class SearchSupportTicketsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, description="Free-text match on the ticket subject.")
    status: Optional[str] = Field(
        default=None, description="open / pending / resolved / closed"
    )
    customer_id: Optional[int] = None
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _validate(self) -> "SearchSupportTicketsInput":
        if self.status is not None and self.status not in _VALID_TICKET_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_TICKET_STATUSES)}"
            )
        return self


class SupportTicketSearchItem(BaseModel):
    ticket_number: str
    customer_id: int
    subject: str
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime


class SearchSupportTicketsOutput(BaseModel):
    count: int
    tickets: list[SupportTicketSearchItem]


def _search_tickets_impl(
    session: Session, inp: SearchSupportTicketsInput
) -> SearchSupportTicketsOutput:
    pattern = f"%{inp.query.strip()}%"
    stmt = (
        select(SupportTicket)
        .where(SupportTicket.subject.ilike(pattern))
        .order_by(SupportTicket.updated_at.desc())
        .limit(inp.limit)
    )
    if inp.status is not None:
        stmt = stmt.where(SupportTicket.status == inp.status)
    if inp.customer_id is not None:
        stmt = stmt.where(SupportTicket.customer_id == inp.customer_id)

    rows = session.execute(stmt).scalars().all()
    items = [
        SupportTicketSearchItem(
            ticket_number=t.ticket_number,
            customer_id=t.customer_id,
            subject=t.subject,
            status=t.status,
            priority=t.priority,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in rows
    ]
    return SearchSupportTicketsOutput(count=len(items), tickets=items)


search_support_tickets = Tool(
    name="search_support_tickets",
    description=(
        "Search support tickets by free-text on the subject, optionally filtered "
        "by status (open/pending/resolved/closed) and customer_id."
    ),
    domain="support",
    input_schema=SearchSupportTicketsInput,
    output_schema=SearchSupportTicketsOutput,
    risk_level="medium",
    read_only=True,
    impl=_search_tickets_impl,
)


# ---------------------------------------------------------------------------
# get_escalation_policy
# ---------------------------------------------------------------------------


_ESCALATION_POLICY: dict[str, dict[str, Any]] = {
    "low": {
        "first_response_sla_hours": 48,
        "resolution_sla_hours": 168,
        "description": "Routine inquiry; handled by frontline support during business hours.",
        "steps": [
            {"step": 1, "role": "frontline support", "channel": "email", "hours_after": 0},
            {"step": 2, "role": "senior support", "channel": "email", "hours_after": 24},
        ],
    },
    "normal": {
        "first_response_sla_hours": 24,
        "resolution_sla_hours": 96,
        "description": "Standard issue; resolved within four business days.",
        "steps": [
            {"step": 1, "role": "frontline support", "channel": "email", "hours_after": 0},
            {"step": 2, "role": "senior support", "channel": "email", "hours_after": 12},
            {"step": 3, "role": "supervisor", "channel": "phone", "hours_after": 48},
        ],
    },
    "high": {
        "first_response_sla_hours": 4,
        "resolution_sla_hours": 24,
        "description": "Material customer impact; same-day resolution targeted.",
        "steps": [
            {"step": 1, "role": "senior support", "channel": "phone", "hours_after": 0},
            {"step": 2, "role": "supervisor", "channel": "phone", "hours_after": 2},
            {"step": 3, "role": "ops lead", "channel": "phone", "hours_after": 8},
        ],
    },
    "urgent": {
        "first_response_sla_hours": 1,
        "resolution_sla_hours": 8,
        "description": "Operational/safety impact; immediate pager-style escalation.",
        "steps": [
            {"step": 1, "role": "ops lead", "channel": "phone", "hours_after": 0},
            {"step": 2, "role": "incident manager", "channel": "phone", "hours_after": 0.5},
            {"step": 3, "role": "engineering director", "channel": "phone", "hours_after": 2},
        ],
    },
}


class GetEscalationPolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: Optional[str] = Field(
        default=None, description="One of: low, normal, high, urgent"
    )
    ticket_number: Optional[str] = Field(
        default=None,
        description="If given, the ticket's priority is read from the DB and used.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetEscalationPolicyInput":
        provided = sum(x is not None for x in (self.priority, self.ticket_number))
        if provided != 1:
            raise ValueError("provide exactly one of priority, ticket_number")
        if self.priority is not None and self.priority not in _VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(_VALID_PRIORITIES)}")
        return self


class EscalationStep(BaseModel):
    step: int
    role: str
    channel: str
    hours_after: float


class GetEscalationPolicyOutput(BaseModel):
    priority: str
    resolved_via: str  # "priority" or "ticket_number"
    ticket_number: Optional[str]
    first_response_sla_hours: int
    resolution_sla_hours: int
    description: str
    steps: list[EscalationStep]


def _escalation_impl(
    session: Session, inp: GetEscalationPolicyInput
) -> GetEscalationPolicyOutput:
    if inp.ticket_number is not None:
        ticket = session.execute(
            select(SupportTicket).where(
                SupportTicket.ticket_number == inp.ticket_number
            )
        ).scalar_one_or_none()
        if ticket is None:
            raise ResourceNotFoundError(f"ticket {inp.ticket_number!r} not found")
        priority = ticket.priority
        resolved_via = "ticket_number"
        ticket_num = ticket.ticket_number
    else:
        priority = inp.priority  # validated above
        resolved_via = "priority"
        ticket_num = None

    policy = _ESCALATION_POLICY.get(priority)
    if policy is None:
        # Fallback (shouldn't happen because seed enforces _VALID_PRIORITIES, but
        # a ticket could in theory carry a different value).
        policy = _ESCALATION_POLICY["normal"]

    return GetEscalationPolicyOutput(
        priority=priority,
        resolved_via=resolved_via,
        ticket_number=ticket_num,
        first_response_sla_hours=policy["first_response_sla_hours"],
        resolution_sla_hours=policy["resolution_sla_hours"],
        description=policy["description"],
        steps=[EscalationStep(**s) for s in policy["steps"]],
    )


get_escalation_policy = Tool(
    name="get_escalation_policy",
    description=(
        "Return the escalation policy (SLA hours, escalation steps) for a "
        "given priority. Lookup by explicit priority or by ticket_number "
        "(which resolves the ticket's priority first)."
    ),
    domain="support",
    input_schema=GetEscalationPolicyInput,
    output_schema=GetEscalationPolicyOutput,
    risk_level="low",
    read_only=True,
    impl=_escalation_impl,
)


# ---------------------------------------------------------------------------
# create_support_ticket_draft  (DRAFT ONLY — no DB write)
# ---------------------------------------------------------------------------


class CreateSupportTicketDraftInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: int
    subject: str = Field(min_length=5, max_length=200)
    priority: str = Field(default="normal")
    description: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _validate(self) -> "CreateSupportTicketDraftInput":
        if self.priority not in _VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(_VALID_PRIORITIES)}")
        return self


class SupportTicketDraft(BaseModel):
    customer_id: int
    subject: str
    priority: str
    description: Optional[str]
    proposed_status: str = "draft_pending_review"


class CreateSupportTicketDraftOutput(BaseModel):
    is_draft: bool  # always True — explicit so the LLM knows nothing was committed
    customer_id: int
    customer_full_name: Optional[str]
    draft_ticket: SupportTicketDraft
    next_steps: str


def _create_ticket_draft_impl(
    session: Session, inp: CreateSupportTicketDraftInput
) -> CreateSupportTicketDraftOutput:
    customer = session.execute(
        select(Customer).where(Customer.id == inp.customer_id)
    ).scalar_one_or_none()
    if customer is None:
        raise ResourceNotFoundError("customer not found")

    draft = SupportTicketDraft(
        customer_id=inp.customer_id,
        subject=inp.subject,
        priority=inp.priority,
        description=inp.description,
    )
    return CreateSupportTicketDraftOutput(
        is_draft=True,
        customer_id=customer.id,
        customer_full_name=customer.full_name,
        draft_ticket=draft,
        next_steps=(
            "This draft has NOT been persisted. The user should review and "
            "explicitly confirm before a ticket is actually created."
        ),
    )


create_support_ticket_draft = Tool(
    name="create_support_ticket_draft",
    description=(
        "Prepare a SUPPORT TICKET DRAFT for a customer. This is DRAFT ONLY — "
        "nothing is written to the DB. Use this when the user asks to open a "
        "ticket; the human-in-the-loop must confirm before any real ticket is "
        "created."
    ),
    domain="support",
    input_schema=CreateSupportTicketDraftInput,
    output_schema=CreateSupportTicketDraftOutput,
    risk_level="medium",
    read_only=True,  # No DB writes — strictly a validation + payload echo.
    impl=_create_ticket_draft_impl,
)
