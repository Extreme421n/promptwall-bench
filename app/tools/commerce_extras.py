"""Phase C2 commerce tools.

* calculate_bundle_price       — read-only; computes a bundle total from
                                  current product prices, supports optional
                                  discount_pct.
* get_commerce_return_status   — focused on the return record itself (not the
                                  refund attached to it); distinct from
                                  get_commerce_refund_status.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    CommerceOrder,
    CommerceReturn,
    Customer,
    Product,
    ProductPrice,
)
from app.tools.base import ResourceNotFoundError, Tool


# ---------------------------------------------------------------------------
# calculate_bundle_price
# ---------------------------------------------------------------------------


class BundleItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sku: str = Field(min_length=1)
    quantity: int = Field(ge=1, le=1000)


class CalculateBundlePriceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[BundleItemInput] = Field(min_length=1, max_length=50)
    discount_pct: float = Field(default=0.0, ge=0.0, le=50.0)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class BundleItemOut(BaseModel):
    sku: str
    product_id: int
    name: str
    quantity: int
    unit_price: Decimal
    line_total: Decimal


class CalculateBundlePriceOutput(BaseModel):
    currency: str
    discount_pct: float
    item_count: int
    subtotal: Decimal
    discount_amount: Decimal
    total: Decimal
    items: list[BundleItemOut]
    notes: Optional[str] = None


def _calculate_bundle_impl(
    session: Session, inp: CalculateBundlePriceInput
) -> CalculateBundlePriceOutput:
    # Look up every product once
    skus = [it.sku.strip() for it in inp.items]
    products = list(
        session.execute(select(Product).where(Product.sku.in_(skus))).scalars().all()
    )
    product_by_sku: dict[str, Product] = {p.sku: p for p in products}

    missing = [s for s in skus if s not in product_by_sku]
    if missing:
        raise ResourceNotFoundError(f"unknown SKUs: {sorted(set(missing))}")

    # Current prices for those products (valid_to IS NULL)
    current_prices = {
        row.product_id: row
        for row in session.execute(
            select(ProductPrice).where(
                ProductPrice.product_id.in_([p.id for p in products]),
                ProductPrice.valid_to.is_(None),
            )
        ).scalars().all()
    }

    items_out: list[BundleItemOut] = []
    subtotal = Decimal("0.00")
    for it in inp.items:
        product = product_by_sku[it.sku.strip()]
        price_row = current_prices.get(product.id)
        if price_row is None:
            raise ResourceNotFoundError(
                f"no current price for SKU {product.sku!r}; price the item before bundling"
            )
        unit_price = price_row.price
        line_total = (unit_price * it.quantity).quantize(Decimal("0.01"))
        subtotal += line_total
        items_out.append(
            BundleItemOut(
                sku=product.sku,
                product_id=product.id,
                name=product.name,
                quantity=it.quantity,
                unit_price=unit_price,
                line_total=line_total,
            )
        )

    discount_amount = (subtotal * Decimal(str(inp.discount_pct)) / Decimal("100")).quantize(
        Decimal("0.01")
    )
    total = (subtotal - discount_amount).quantize(Decimal("0.01"))

    notes = None
    if inp.currency.upper() != "USD":
        notes = (
            f"All product prices in this catalog are stored in USD; "
            f"requested currency {inp.currency!r} was passed through but no "
            "FX conversion was applied."
        )

    return CalculateBundlePriceOutput(
        currency=inp.currency.upper(),
        discount_pct=inp.discount_pct,
        item_count=sum(i.quantity for i in inp.items),
        subtotal=subtotal.quantize(Decimal("0.01")),
        discount_amount=discount_amount,
        total=total,
        items=items_out,
        notes=notes,
    )


calculate_bundle_price = Tool(
    name="calculate_bundle_price",
    description=(
        "Calculate the total price for a basket of products. Input is a list "
        "of {sku, quantity}; the tool looks up each product's current price, "
        "applies an optional discount_pct (0–50%), and returns subtotal / "
        "discount / total with per-line detail."
    ),
    domain="commerce",
    input_schema=CalculateBundlePriceInput,
    output_schema=CalculateBundlePriceOutput,
    risk_level="low",
    read_only=True,
    impl=_calculate_bundle_impl,
)


# ---------------------------------------------------------------------------
# get_commerce_return_status
# ---------------------------------------------------------------------------


class GetCommerceReturnStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_number: Optional[str] = None
    customer_id: Optional[int] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "GetCommerceReturnStatusInput":
        provided = sum(x is not None for x in (self.order_number, self.customer_id))
        if provided != 1:
            raise ValueError("provide exactly one of order_number, customer_id")
        return self


class CommerceReturnItemOut(BaseModel):
    return_id: int
    order_number: str
    status: str
    reason: Optional[str]
    created_at: datetime
    updated_at: datetime


class GetCommerceReturnStatusOutput(BaseModel):
    count: int
    returns: list[CommerceReturnItemOut]


def _return_status_impl(
    session: Session, inp: GetCommerceReturnStatusInput
) -> GetCommerceReturnStatusOutput:
    if inp.order_number is not None:
        order = session.execute(
            select(CommerceOrder).where(CommerceOrder.order_number == inp.order_number)
        ).scalar_one_or_none()
        if order is None:
            raise ResourceNotFoundError(
                f"commerce order {inp.order_number!r} not found"
            )
        rows = session.execute(
            select(CommerceReturn, CommerceOrder.order_number)
            .join(CommerceOrder, CommerceOrder.id == CommerceReturn.order_id)
            .where(CommerceReturn.order_id == order.id)
            .order_by(CommerceReturn.updated_at.desc())
        ).all()
    else:
        exists = session.execute(
            select(Customer.id).where(Customer.id == inp.customer_id)
        ).scalar_one_or_none()
        if exists is None:
            raise ResourceNotFoundError("customer not found")
        rows = session.execute(
            select(CommerceReturn, CommerceOrder.order_number)
            .join(CommerceOrder, CommerceOrder.id == CommerceReturn.order_id)
            .where(CommerceOrder.customer_id == inp.customer_id)
            .order_by(CommerceReturn.updated_at.desc())
            .limit(50)
        ).all()

    items = [
        CommerceReturnItemOut(
            return_id=r.id,
            order_number=order_number,
            status=r.status,
            reason=r.reason,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r, order_number in rows
    ]
    return GetCommerceReturnStatusOutput(count=len(items), returns=items)


get_commerce_return_status = Tool(
    name="get_commerce_return_status",
    description=(
        "Look up the RETURN lifecycle (requested / approved / rejected / "
        "completed) for a commerce order. Distinct from "
        "get_commerce_refund_status, which focuses on the refund payment "
        "attached to a return. Lookup by order_number or customer_id."
    ),
    domain="commerce",
    input_schema=GetCommerceReturnStatusInput,
    output_schema=GetCommerceReturnStatusOutput,
    risk_level="medium",
    read_only=True,
    impl=_return_status_impl,
)
