"""Commerce order/return/shipment tools (Phase C1).

get_commerce_order_status   — lookup orders by number or customer id.
get_commerce_refund_status  — lookup commerce returns + refunds.
get_shipment_status         — lookup shipments by tracking or order number.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.models import (
    CommerceOrder,
    CommerceOrderItem,
    CommerceRefund,
    CommerceReturn,
    Customer,
    Shipment,
)
from app.tools.base import ResourceNotFoundError, Tool


# ---------------------------------------------------------------------------
# get_commerce_order_status
# ---------------------------------------------------------------------------


class GetCommerceOrderStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_number: Optional[str] = Field(default=None)
    customer_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetCommerceOrderStatusInput":
        provided = sum(x is not None for x in (self.order_number, self.customer_id))
        if provided != 1:
            raise ValueError("provide exactly one of order_number, customer_id")
        return self


class CommerceOrderItemOut(BaseModel):
    order_number: str
    customer_id: int
    status: str
    total_amount: Decimal
    currency: str
    created_at: datetime
    item_count: int
    latest_shipment_status: Optional[str]
    latest_tracking_number: Optional[str]


class GetCommerceOrderStatusOutput(BaseModel):
    count: int
    orders: list[CommerceOrderItemOut]


def _order_status_impl(
    session: Session, inp: GetCommerceOrderStatusInput
) -> GetCommerceOrderStatusOutput:
    if inp.order_number is not None:
        orders = session.execute(
            select(CommerceOrder).where(CommerceOrder.order_number == inp.order_number)
        ).scalars().all()
        if not orders:
            raise ResourceNotFoundError(
                f"commerce order {inp.order_number!r} not found"
            )
    else:
        exists = session.execute(
            select(Customer.id).where(Customer.id == inp.customer_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("customer not found")
        orders = session.execute(
            select(CommerceOrder)
            .where(CommerceOrder.customer_id == inp.customer_id)
            .order_by(CommerceOrder.created_at.desc())
            .limit(50)
        ).scalars().all()

    items: list[CommerceOrderItemOut] = []
    for o in orders:
        item_count = session.execute(
            select(func.count())
            .select_from(CommerceOrderItem)
            .where(CommerceOrderItem.order_id == o.id)
        ).scalar_one()
        latest_ship = session.execute(
            select(Shipment)
            .where(Shipment.order_id == o.id)
            .order_by(desc(Shipment.updated_at))
            .limit(1)
        ).scalar_one_or_none()
        items.append(
            CommerceOrderItemOut(
                order_number=o.order_number,
                customer_id=o.customer_id,
                status=o.status,
                total_amount=o.total_amount,
                currency=o.currency,
                created_at=o.created_at,
                item_count=int(item_count),
                latest_shipment_status=(latest_ship.status if latest_ship else None),
                latest_tracking_number=(
                    latest_ship.tracking_number if latest_ship else None
                ),
            )
        )
    return GetCommerceOrderStatusOutput(count=len(items), orders=items)


get_commerce_order_status = Tool(
    name="get_commerce_order_status",
    description=(
        "Look up commerce order(s) by order_number (single order) or "
        "customer_id (all of that customer's orders, newest first, up to 50). "
        "Each row includes an item count and the latest shipment's status."
    ),
    domain="commerce",
    input_schema=GetCommerceOrderStatusInput,
    output_schema=GetCommerceOrderStatusOutput,
    risk_level="medium",
    read_only=True,
    impl=_order_status_impl,
)


# ---------------------------------------------------------------------------
# get_commerce_refund_status
# ---------------------------------------------------------------------------


class GetCommerceRefundStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_number: Optional[str] = Field(default=None)
    customer_id: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetCommerceRefundStatusInput":
        provided = sum(x is not None for x in (self.order_number, self.customer_id))
        if provided != 1:
            raise ValueError("provide exactly one of order_number, customer_id")
        return self


class CommerceRefundItemOut(BaseModel):
    order_number: str
    return_id: int
    return_status: str
    return_reason: Optional[str]
    refund_status: Optional[str]
    refund_amount: Optional[Decimal]
    expected_resolution_date: Optional[date]
    updated_at: datetime


class GetCommerceRefundStatusOutput(BaseModel):
    count: int
    refunds: list[CommerceRefundItemOut]


def _refund_status_impl(
    session: Session, inp: GetCommerceRefundStatusInput
) -> GetCommerceRefundStatusOutput:
    if inp.order_number is not None:
        order = session.execute(
            select(CommerceOrder).where(
                CommerceOrder.order_number == inp.order_number
            )
        ).scalar_one_or_none()
        if order is None:
            raise ResourceNotFoundError(
                f"commerce order {inp.order_number!r} not found"
            )
        order_ids = [order.id]
        order_numbers = {order.id: order.order_number}
    else:
        exists = session.execute(
            select(Customer.id).where(Customer.id == inp.customer_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("customer not found")
        rows = session.execute(
            select(CommerceOrder.id, CommerceOrder.order_number).where(
                CommerceOrder.customer_id == inp.customer_id
            )
        ).all()
        order_ids = [r[0] for r in rows]
        order_numbers = {r[0]: r[1] for r in rows}
        if not order_ids:
            return GetCommerceRefundStatusOutput(count=0, refunds=[])

    returns = session.execute(
        select(CommerceReturn).where(CommerceReturn.order_id.in_(order_ids))
    ).scalars().all()

    items: list[CommerceRefundItemOut] = []
    for r in returns:
        latest_refund = session.execute(
            select(CommerceRefund)
            .where(CommerceRefund.return_id == r.id)
            .order_by(CommerceRefund.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        items.append(
            CommerceRefundItemOut(
                order_number=order_numbers[r.order_id],
                return_id=r.id,
                return_status=r.status,
                return_reason=r.reason,
                refund_status=(latest_refund.refund_status if latest_refund else None),
                refund_amount=(latest_refund.refund_amount if latest_refund else None),
                expected_resolution_date=(
                    latest_refund.expected_resolution_date if latest_refund else None
                ),
                updated_at=(latest_refund.updated_at if latest_refund else r.updated_at),
            )
        )
    return GetCommerceRefundStatusOutput(count=len(items), refunds=items)


get_commerce_refund_status = Tool(
    name="get_commerce_refund_status",
    description=(
        "Look up commerce return and refund status by order_number or "
        "customer_id. Each row pairs the return with the latest associated "
        "refund. NOT for airline refunds — use get_refund_status instead."
    ),
    domain="commerce",
    input_schema=GetCommerceRefundStatusInput,
    output_schema=GetCommerceRefundStatusOutput,
    risk_level="medium",
    read_only=True,
    impl=_refund_status_impl,
)


# ---------------------------------------------------------------------------
# get_shipment_status
# ---------------------------------------------------------------------------


class GetShipmentStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_number: Optional[str] = Field(default=None)
    order_number: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetShipmentStatusInput":
        provided = sum(x is not None for x in (self.tracking_number, self.order_number))
        if provided != 1:
            raise ValueError("provide exactly one of tracking_number, order_number")
        return self


class ShipmentItemOut(BaseModel):
    order_number: str
    tracking_number: str
    carrier: str
    status: str
    estimated_delivery: Optional[datetime]
    updated_at: datetime


class GetShipmentStatusOutput(BaseModel):
    count: int
    shipments: list[ShipmentItemOut]


def _shipment_impl(
    session: Session, inp: GetShipmentStatusInput
) -> GetShipmentStatusOutput:
    stmt = (
        select(Shipment, CommerceOrder.order_number)
        .join(CommerceOrder, Shipment.order_id == CommerceOrder.id)
        .order_by(Shipment.updated_at.desc())
        .limit(25)
    )
    if inp.tracking_number is not None:
        stmt = stmt.where(Shipment.tracking_number == inp.tracking_number.strip())
    else:
        stmt = stmt.where(CommerceOrder.order_number == inp.order_number.strip())

    rows = session.execute(stmt).all()
    if not rows:
        ref = inp.tracking_number or inp.order_number
        raise ResourceNotFoundError(f"no shipments found for {ref!r}")

    items = [
        ShipmentItemOut(
            order_number=order_number,
            tracking_number=s.tracking_number,
            carrier=s.carrier,
            status=s.status,
            estimated_delivery=s.estimated_delivery,
            updated_at=s.updated_at,
        )
        for s, order_number in rows
    ]
    return GetShipmentStatusOutput(count=len(items), shipments=items)


get_shipment_status = Tool(
    name="get_shipment_status",
    description=(
        "Look up shipment(s) by tracking_number (single shipment) or "
        "order_number (all shipments for that order, newest first)."
    ),
    domain="commerce",
    input_schema=GetShipmentStatusInput,
    output_schema=GetShipmentStatusOutput,
    risk_level="low",
    read_only=True,
    impl=_shipment_impl,
)
