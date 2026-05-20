"""get_invoice_status tool (Phase C1)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Invoice, Organization
from app.tools.base import ResourceNotFoundError, Tool


class GetInvoiceStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: Optional[str] = Field(default=None)
    organization_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetInvoiceStatusInput":
        provided = sum(x is not None for x in (self.invoice_number, self.organization_id))
        if provided != 1:
            raise ValueError("provide exactly one of invoice_number, organization_id")
        return self


class InvoiceItemOut(BaseModel):
    invoice_number: str
    organization_id: int
    status: str
    total_amount: Decimal
    currency: str
    issued_at: datetime
    due_at: datetime
    paid_at: Optional[datetime]


class GetInvoiceStatusOutput(BaseModel):
    count: int
    invoices: list[InvoiceItemOut]


def _impl(session: Session, inp: GetInvoiceStatusInput) -> GetInvoiceStatusOutput:
    stmt = select(Invoice)
    if inp.invoice_number is not None:
        stmt = stmt.where(Invoice.invoice_number == inp.invoice_number)
    else:
        # Validate org first so a wrong id 404s cleanly.
        exists = session.execute(
            select(Organization.id).where(Organization.id == inp.organization_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("organization not found")
        stmt = stmt.where(Invoice.organization_id == inp.organization_id)
    stmt = stmt.order_by(Invoice.issued_at.desc()).limit(50)

    rows = session.execute(stmt).scalars().all()
    if inp.invoice_number is not None and not rows:
        raise ResourceNotFoundError(f"invoice {inp.invoice_number!r} not found")

    items = [
        InvoiceItemOut(
            invoice_number=r.invoice_number,
            organization_id=r.organization_id,
            status=r.status,
            total_amount=r.total_amount,
            currency=r.currency,
            issued_at=r.issued_at,
            due_at=r.due_at,
            paid_at=r.paid_at,
        )
        for r in rows
    ]
    return GetInvoiceStatusOutput(count=len(items), invoices=items)


get_invoice_status = Tool(
    name="get_invoice_status",
    description=(
        "Look up SaaS invoices by invoice_number (single invoice) or "
        "organization_id (all of that org's invoices, newest first, up to 50)."
    ),
    domain="saas",
    input_schema=GetInvoiceStatusInput,
    output_schema=GetInvoiceStatusOutput,
    risk_level="medium",
    read_only=True,
    impl=_impl,
)
