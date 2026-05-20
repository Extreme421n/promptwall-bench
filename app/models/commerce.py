"""Commerce / orders domain models (Phase B2).

Adds a third business domain alongside airline + SaaS so the chatbot
benchmark has overlapping vocabulary in more dimensions:

* ``order`` may be an airline booking *or* a commerce order
  (``commerce_orders``).
* ``refund`` may be an airline refund (``refunds``), a SaaS credit
  (none yet — future), or a commerce refund (``commerce_refunds``).
* ``status`` now covers a third axis: order, shipment, return, refund.
* ``shipment`` is commerce-specific and shouldn't be confused with a
  flight movement.
* ``product availability`` requires reading ``product_inventory`` —
  another ``available`` synonym alongside flight/seat availability.

Phase B2 adds schema + seed only. No tools, no chatbot changes.
"""

from __future__ import annotations

from datetime import date as _date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.crm import Customer


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class ProductCategory(Base):
    __tablename__ = "product_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("product_categories.id", ondelete="SET NULL"), nullable=True
    )

    children: Mapped[list["ProductCategory"]] = relationship(
        "ProductCategory",
        backref="parent",
        remote_side="ProductCategory.id",
    )

    __table_args__ = (Index("ix_product_categories_parent_id", "parent_id"),)


class Product(Base, TimestampMixin):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category_id: Mapped[int] = mapped_column(
        ForeignKey("product_categories.id", ondelete="RESTRICT"), nullable=False
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    brand: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    category: Mapped["ProductCategory"] = relationship()
    attributes: Mapped[list["ProductAttribute"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    prices: Mapped[list["ProductPrice"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    inventory: Mapped[list["ProductInventory"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_products_category_id", "category_id"),
        Index("ix_products_brand", "brand"),
        Index("ix_products_is_active", "is_active"),
    )


class ProductAttribute(Base):
    __tablename__ = "product_attributes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    attribute_name: Mapped[str] = mapped_column(String(80), nullable=False)
    attribute_value: Mapped[str] = mapped_column(String(200), nullable=False)

    product: Mapped["Product"] = relationship(back_populates="attributes")

    __table_args__ = (
        Index("ix_product_attributes_product_id", "product_id"),
        Index("ix_product_attributes_name", "attribute_name"),
    )


class ProductPrice(Base):
    __tablename__ = "product_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    product: Mapped["Product"] = relationship(back_populates="prices")

    __table_args__ = (
        Index("ix_product_prices_product_id", "product_id"),
        Index("ix_product_prices_valid_from", "valid_from"),
    )


# ---------------------------------------------------------------------------
# Warehouses + inventory
# ---------------------------------------------------------------------------


class Warehouse(Base):
    __tablename__ = "warehouses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    city: Mapped[str] = mapped_column(String(80), nullable=False)
    country: Mapped[str] = mapped_column(String(80), nullable=False)


class ProductInventory(Base):
    __tablename__ = "product_inventory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE"), nullable=False
    )
    warehouse_id: Mapped[int] = mapped_column(
        ForeignKey("warehouses.id", ondelete="CASCADE"), nullable=False
    )
    quantity_available: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    product: Mapped["Product"] = relationship(back_populates="inventory")
    warehouse: Mapped["Warehouse"] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "product_id", "warehouse_id", name="uq_product_inventory_product_warehouse"
        ),
        Index("ix_product_inventory_product_id", "product_id"),
        Index("ix_product_inventory_warehouse_id", "warehouse_id"),
    )


# ---------------------------------------------------------------------------
# Orders, items, shipments
# ---------------------------------------------------------------------------


class CommerceOrder(Base, TimestampMixin):
    __tablename__ = "commerce_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_number: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="placed")
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")

    items: Mapped[list["CommerceOrderItem"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )
    shipments: Mapped[list["Shipment"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )
    returns: Mapped[list["CommerceReturn"]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_commerce_orders_customer_id", "customer_id"),
        Index("ix_commerce_orders_status", "status"),
    )


class CommerceOrderItem(Base):
    __tablename__ = "commerce_order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("commerce_orders.id", ondelete="CASCADE"), nullable=False
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", ondelete="RESTRICT"), nullable=False
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)

    order: Mapped["CommerceOrder"] = relationship(back_populates="items")

    __table_args__ = (
        Index("ix_commerce_order_items_order_id", "order_id"),
        Index("ix_commerce_order_items_product_id", "product_id"),
    )


class Shipment(Base):
    __tablename__ = "shipments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("commerce_orders.id", ondelete="CASCADE"), nullable=False
    )
    tracking_number: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    carrier: Mapped[str] = mapped_column(String(40), nullable=False)
    estimated_delivery: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    order: Mapped["CommerceOrder"] = relationship(back_populates="shipments")

    __table_args__ = (
        Index("ix_shipments_order_id", "order_id"),
        Index("ix_shipments_status", "status"),
    )


# ---------------------------------------------------------------------------
# Returns + refunds (commerce)
# ---------------------------------------------------------------------------


class CommerceReturn(Base, TimestampMixin):
    __tablename__ = "commerce_returns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("commerce_orders.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="requested")
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    order: Mapped["CommerceOrder"] = relationship(back_populates="returns")
    refunds: Mapped[list["CommerceRefund"]] = relationship(
        back_populates="return_", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_commerce_returns_order_id", "order_id"),
        Index("ix_commerce_returns_status", "status"),
    )


class CommerceRefund(Base):
    """Commerce refund — distinct from airline ``refunds`` table.

    The two tables exist deliberately so the chatbot has to pick the right
    refund tool based on whether the user is asking about a flight booking
    or a commerce order.
    """

    __tablename__ = "commerce_refunds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    return_id: Mapped[int] = mapped_column(
        ForeignKey("commerce_returns.id", ondelete="CASCADE"), nullable=False
    )
    refund_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    refund_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    expected_resolution_date: Mapped[Optional[_date]] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # `return` is a Python keyword, so the Python attribute is `return_`.
    return_: Mapped["CommerceReturn"] = relationship(back_populates="refunds")

    __table_args__ = (
        Index("ix_commerce_refunds_return_id", "return_id"),
        Index("ix_commerce_refunds_status", "refund_status"),
    )
