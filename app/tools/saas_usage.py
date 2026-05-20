"""SaaS usage tools (Phase C1): calculate_usage_overage + get_api_usage_summary."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import ApiUsageDaily, Organization, Plan, Subscription
from app.tools.base import ResourceNotFoundError, Tool


# ---------------------------------------------------------------------------
# Shared validator helper
# ---------------------------------------------------------------------------


def _validate_org_exists(session: Session, organization_id: int) -> None:
    exists = session.execute(
        select(Organization.id).where(Organization.id == organization_id)
    ).scalar_one_or_none()
    if exists is None:
        raise ResourceNotFoundError("organization not found")


# ---------------------------------------------------------------------------
# calculate_usage_overage
# ---------------------------------------------------------------------------


class CalculateUsageOverageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: int = Field(description="Internal organization id")
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def _validate_range(self) -> "CalculateUsageOverageInput":
        if self.date_to < self.date_from:
            raise ValueError("date_to must be on or after date_from")
        return self


class CalculateUsageOverageOutput(BaseModel):
    organization_id: int
    plan_name: Optional[str]
    date_from: date
    date_to: date
    days_in_range: int
    total_api_calls: int
    included_quota: int
    overage_calls: int
    overage_price_per_1000_calls: Decimal
    estimated_overage_charge_usd: Decimal


def _overage_impl(
    session: Session, inp: CalculateUsageOverageInput
) -> CalculateUsageOverageOutput:
    _validate_org_exists(session, inp.organization_id)

    # Find this org's most recent subscription (and its plan).
    sub_row = session.execute(
        select(Subscription, Plan)
        .join(Plan, Subscription.plan_id == Plan.id)
        .where(Subscription.organization_id == inp.organization_id)
        .order_by(Subscription.id.desc())
        .limit(1)
    ).one_or_none()
    plan: Optional[Plan] = sub_row[1] if sub_row else None

    total_calls = session.execute(
        select(func.coalesce(func.sum(ApiUsageDaily.api_calls), 0))
        .where(
            ApiUsageDaily.organization_id == inp.organization_id,
            ApiUsageDaily.date >= inp.date_from,
            ApiUsageDaily.date <= inp.date_to,
        )
    ).scalar_one()
    total_calls = int(total_calls)

    days_in_range = (inp.date_to - inp.date_from).days + 1

    if plan is None:
        included = 0
        rate = Decimal("0.0000")
    else:
        # Treat plan limit as monthly (30 days) and pro-rate for the range.
        included = int(plan.included_api_calls * (days_in_range / 30.0))
        rate = plan.overage_price_per_1000_calls

    overage = max(0, total_calls - included)
    charge = (Decimal(overage) * rate / Decimal("1000")).quantize(Decimal("0.01"))

    return CalculateUsageOverageOutput(
        organization_id=inp.organization_id,
        plan_name=plan.name if plan else None,
        date_from=inp.date_from,
        date_to=inp.date_to,
        days_in_range=days_in_range,
        total_api_calls=total_calls,
        included_quota=included,
        overage_calls=overage,
        overage_price_per_1000_calls=rate,
        estimated_overage_charge_usd=charge,
    )


calculate_usage_overage = Tool(
    name="calculate_usage_overage",
    description=(
        "Compute an estimated overage charge for an organization over a date "
        "range. Uses the org's current plan as the included quota (pro-rated "
        "to the range, treating the plan as monthly) and the plan's overage rate."
    ),
    domain="saas",
    input_schema=CalculateUsageOverageInput,
    output_schema=CalculateUsageOverageOutput,
    risk_level="medium",
    read_only=True,
    impl=_overage_impl,
)


# ---------------------------------------------------------------------------
# get_api_usage_summary
# ---------------------------------------------------------------------------


class GetApiUsageSummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization_id: int
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def _validate_range(self) -> "GetApiUsageSummaryInput":
        if self.date_to < self.date_from:
            raise ValueError("date_to must be on or after date_from")
        return self


class GetApiUsageSummaryOutput(BaseModel):
    organization_id: int
    date_from: date
    date_to: date
    days_with_data: int
    total_calls: int
    successful_calls: int
    failed_calls: int
    success_rate: float  # 0..1


def _usage_summary_impl(
    session: Session, inp: GetApiUsageSummaryInput
) -> GetApiUsageSummaryOutput:
    _validate_org_exists(session, inp.organization_id)

    row = session.execute(
        select(
            func.count().label("days"),
            func.coalesce(func.sum(ApiUsageDaily.api_calls), 0),
            func.coalesce(func.sum(ApiUsageDaily.successful_calls), 0),
            func.coalesce(func.sum(ApiUsageDaily.failed_calls), 0),
        ).where(
            ApiUsageDaily.organization_id == inp.organization_id,
            ApiUsageDaily.date >= inp.date_from,
            ApiUsageDaily.date <= inp.date_to,
        )
    ).one()
    days_with_data = int(row[0] or 0)
    total = int(row[1] or 0)
    succ = int(row[2] or 0)
    failed = int(row[3] or 0)
    rate = (succ / total) if total > 0 else 0.0

    return GetApiUsageSummaryOutput(
        organization_id=inp.organization_id,
        date_from=inp.date_from,
        date_to=inp.date_to,
        days_with_data=days_with_data,
        total_calls=total,
        successful_calls=succ,
        failed_calls=failed,
        success_rate=round(rate, 4),
    )


get_api_usage_summary = Tool(
    name="get_api_usage_summary",
    description=(
        "Aggregate API usage for an organization over a date range. Returns "
        "total calls, successful calls, failed calls and a success rate."
    ),
    domain="saas",
    input_schema=GetApiUsageSummaryInput,
    output_schema=GetApiUsageSummaryOutput,
    risk_level="low",
    read_only=True,
    impl=_usage_summary_impl,
)
