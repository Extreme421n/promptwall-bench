"""get_baggage_policy tool."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import BaggageRule
from app.tools.base import ResourceNotFoundError, Tool

_VALID_CABINS = {"economy", "premium_economy", "business", "first"}
_VALID_ROUTES = {"domestic", "intra-continental", "international", "ultra-long-haul"}


class GetBaggagePolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cabin_class: str = Field(description="economy, premium_economy, business, or first")
    route_type: Optional[str] = Field(
        default=None,
        description="domestic, intra-continental, international, or ultra-long-haul",
    )


class BaggagePolicyItem(BaseModel):
    route_type: str
    cabin_class: str
    checked_bag_kg: int
    cabin_bag_kg: int
    policy_text: str
    effective_from: date


class GetBaggagePolicyOutput(BaseModel):
    count: int
    policies: list[BaggagePolicyItem]


def _impl(session: Session, inp: GetBaggagePolicyInput) -> GetBaggagePolicyOutput:
    if inp.cabin_class not in _VALID_CABINS:
        raise ResourceNotFoundError(
            f"unknown cabin_class {inp.cabin_class!r}; "
            f"expected one of {sorted(_VALID_CABINS)}"
        )
    if inp.route_type is not None and inp.route_type not in _VALID_ROUTES:
        raise ResourceNotFoundError(
            f"unknown route_type {inp.route_type!r}; expected one of {sorted(_VALID_ROUTES)}"
        )

    # Return the most-recent (latest effective_from) rule per route_type for the
    # requested cabin. If route_type is specified, only that combination is
    # returned.
    stmt = select(BaggageRule).where(BaggageRule.cabin_class == inp.cabin_class)
    if inp.route_type is not None:
        stmt = stmt.where(BaggageRule.route_type == inp.route_type)
    stmt = stmt.order_by(BaggageRule.effective_from.desc())

    rows = session.execute(stmt).scalars().all()
    if not rows:
        raise ResourceNotFoundError("no baggage policy matches that combination")

    latest_per_route: dict[str, BaggageRule] = {}
    for r in rows:
        latest_per_route.setdefault(r.route_type, r)

    items = [
        BaggagePolicyItem(
            route_type=r.route_type,
            cabin_class=r.cabin_class,
            checked_bag_kg=r.checked_bag_kg,
            cabin_bag_kg=r.cabin_bag_kg,
            policy_text=r.policy_text,
            effective_from=r.effective_from,
        )
        for r in latest_per_route.values()
    ]
    items.sort(key=lambda x: x.route_type)
    return GetBaggagePolicyOutput(count=len(items), policies=items)


get_baggage_policy = Tool(
    name="get_baggage_policy",
    description=(
        "Get the latest baggage policy for a cabin class. If route_type is omitted, "
        "returns the latest rule per route_type (domestic, intra-continental, "
        "international, ultra-long-haul)."
    ),
    domain="airline",
    input_schema=GetBaggagePolicyInput,
    output_schema=GetBaggagePolicyOutput,
    risk_level="low",
    read_only=True,
    impl=_impl,
)
