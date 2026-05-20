"""add commerce/orders domain

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-19
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

from app.models import (
    CommerceOrder,
    CommerceOrderItem,
    CommerceRefund,
    CommerceReturn,
    Product,
    ProductAttribute,
    ProductCategory,
    ProductInventory,
    ProductPrice,
    Shipment,
    Warehouse,
)

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# FK-ordered. Creation goes parent → child; drop reverses.
_CREATE_ORDER = (
    ProductCategory,      # self-FK on parent_id
    Product,              # FK -> product_categories
    ProductAttribute,     # FK -> products
    ProductPrice,         # FK -> products
    Warehouse,
    ProductInventory,     # FK -> products, warehouses
    CommerceOrder,        # FK -> customers
    CommerceOrderItem,    # FK -> commerce_orders, products
    Shipment,             # FK -> commerce_orders
    CommerceReturn,       # FK -> commerce_orders
    CommerceRefund,       # FK -> commerce_returns
)


def upgrade() -> None:
    bind = op.get_bind()
    for model in _CREATE_ORDER:
        model.__table__.create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for model in reversed(_CREATE_ORDER):
        model.__table__.drop(bind=bind, checkfirst=True)
