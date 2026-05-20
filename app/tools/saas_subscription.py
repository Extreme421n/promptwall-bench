"""SaaS subscription + plan-limit tools (Phase C1).

get_subscription_status — by customer_id (all orgs they're in) or organization_id.
get_plan_limits         — by plan_name or organization_id (resolves current plan).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import (
    CustomerOrganization,
    Organization,
    Plan,
    Subscription,
)
from app.tools.base import ResourceNotFoundError, Tool


# ---------------------------------------------------------------------------
# get_subscription_status
# ---------------------------------------------------------------------------


class GetSubscriptionStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: Optional[int] = Field(default=None)
    organization_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetSubscriptionStatusInput":
        provided = sum(x is not None for x in (self.customer_id, self.organization_id))
        if provided != 1:
            raise ValueError("provide exactly one of customer_id, organization_id")
        return self


class SubscriptionItem(BaseModel):
    organization_id: int
    organization_name: str
    plan_name: str
    plan_tier: str
    status: str
    started_at: datetime
    renews_at: datetime
    canceled_at: Optional[datetime]


class GetSubscriptionStatusOutput(BaseModel):
    count: int
    subscriptions: list[SubscriptionItem]


def _subscription_status_impl(
    session: Session, inp: GetSubscriptionStatusInput
) -> GetSubscriptionStatusOutput:
    if inp.organization_id is not None:
        # Validate org exists so we can return a clean not-found.
        exists = session.execute(
            select(Organization.id).where(Organization.id == inp.organization_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("organization not found")
        org_ids = [inp.organization_id]
    else:
        # All organizations the customer is a member of.
        org_ids = list(
            session.execute(
                select(CustomerOrganization.organization_id).where(
                    CustomerOrganization.customer_id == inp.customer_id
                )
            ).scalars().all()
        )
        if not org_ids:
            # We don't 404 here — a customer with no orgs is a legitimate result.
            return GetSubscriptionStatusOutput(count=0, subscriptions=[])

    rows = session.execute(
        select(Subscription, Organization.name, Plan.name, Plan.tier)
        .join(Organization, Subscription.organization_id == Organization.id)
        .join(Plan, Subscription.plan_id == Plan.id)
        .where(Subscription.organization_id.in_(org_ids))
        .order_by(Subscription.id)
    ).all()

    items = [
        SubscriptionItem(
            organization_id=sub.organization_id,
            organization_name=org_name,
            plan_name=plan_name,
            plan_tier=plan_tier,
            status=sub.status,
            started_at=sub.started_at,
            renews_at=sub.renews_at,
            canceled_at=sub.canceled_at,
        )
        for sub, org_name, plan_name, plan_tier in rows
    ]
    return GetSubscriptionStatusOutput(count=len(items), subscriptions=items)


get_subscription_status = Tool(
    name="get_subscription_status",
    description=(
        "Look up SaaS subscription status by customer_id (returns subscriptions "
        "for every organization the customer belongs to) or organization_id."
    ),
    domain="saas",
    input_schema=GetSubscriptionStatusInput,
    output_schema=GetSubscriptionStatusOutput,
    risk_level="medium",
    read_only=True,
    impl=_subscription_status_impl,
)


# ---------------------------------------------------------------------------
# get_plan_limits
# ---------------------------------------------------------------------------


class GetPlanLimitsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_name: Optional[str] = Field(default=None)
    organization_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetPlanLimitsInput":
        provided = sum(x is not None for x in (self.plan_name, self.organization_id))
        if provided != 1:
            raise ValueError("provide exactly one of plan_name, organization_id")
        return self


class GetPlanLimitsOutput(BaseModel):
    plan_name: str
    tier: str
    monthly_price: Decimal
    currency: str
    included_seats: int
    included_api_calls: int
    overage_price_per_1000_calls: Decimal
    is_active: bool
    resolved_via: str  # "plan_name" or "organization_id"


def _plan_limits_impl(session: Session, inp: GetPlanLimitsInput) -> GetPlanLimitsOutput:
    if inp.plan_name is not None:
        plan = session.execute(
            select(Plan).where(Plan.name.ilike(inp.plan_name.strip()))
        ).scalar_one_or_none()
        if plan is None:
            raise ResourceNotFoundError(f"plan {inp.plan_name!r} not found")
        return _plan_to_output(plan, resolved_via="plan_name")

    # by organization_id
    exists = session.execute(
        select(Organization.id).where(Organization.id == inp.organization_id)
    ).scalar_one_or_none()
    if exists is None:
        raise ResourceNotFoundError("organization not found")
    sub = session.execute(
        select(Subscription)
        .options(selectinload(Subscription.plan))
        .where(Subscription.organization_id == inp.organization_id)
        .order_by(Subscription.id.desc())
        .limit(1)
    ).scalar_one_or_none()
    if sub is None:
        raise ResourceNotFoundError("organization has no subscription")
    return _plan_to_output(sub.plan, resolved_via="organization_id")


def _plan_to_output(plan: Plan, *, resolved_via: str) -> GetPlanLimitsOutput:
    return GetPlanLimitsOutput(
        plan_name=plan.name,
        tier=plan.tier,
        monthly_price=plan.monthly_price,
        currency="USD",
        included_seats=plan.included_seats,
        included_api_calls=plan.included_api_calls,
        overage_price_per_1000_calls=plan.overage_price_per_1000_calls,
        is_active=plan.is_active,
        resolved_via=resolved_via,
    )


get_plan_limits = Tool(
    name="get_plan_limits",
    description=(
        "Return SaaS plan limits and pricing. Identify the plan by plan_name "
        "(e.g. 'Pro') or by organization_id (resolves to that org's current plan)."
    ),
    domain="saas",
    input_schema=GetPlanLimitsInput,
    output_schema=GetPlanLimitsOutput,
    risk_level="low",
    read_only=True,
    impl=_plan_limits_impl,
)
