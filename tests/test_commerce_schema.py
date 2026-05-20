"""Tests for the commerce/orders schema (Phase B2)."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    CommerceOrder,
    CommerceOrderItem,
    CommerceRefund,
    CommerceReturn,
    Customer,
    Product,
    ProductAttribute,
    ProductCategory,
    ProductInventory,
    ProductPrice,
    Shipment,
    Warehouse,
)
from app.seed import SCALES


_COMMERCE_KEYS = (
    "product_categories",
    "products",
    "product_attributes",
    "product_prices",
    "warehouses",
    "product_inventory",
    "commerce_orders",
    "commerce_order_items",
    "shipments",
    "commerce_returns",
    "commerce_refunds",
)


# ---------------------------------------------------------------------------
# SCALES integrity
# ---------------------------------------------------------------------------


def test_commerce_keys_present_in_every_preset() -> None:
    for preset in ("small", "medium", "large"):
        for key in _COMMERCE_KEYS:
            assert key in SCALES[preset], f"missing {key!r} from {preset!r}"


def test_commerce_counts_monotonic_across_scales() -> None:
    for key in _COMMERCE_KEYS:
        assert SCALES["large"][key] >= SCALES["medium"][key], key
        assert SCALES["medium"][key] >= SCALES["small"][key], key


def test_small_commerce_counts_match_phase_b2_spec() -> None:
    s = SCALES["small"]
    assert s["product_categories"] == 20
    assert s["products"] == 100
    assert s["product_attributes"] == 300
    assert s["product_prices"] == 200
    assert s["warehouses"] == 5
    assert s["product_inventory"] == 500
    assert s["commerce_orders"] == 300
    assert s["commerce_order_items"] == 900
    assert s["shipments"] == 280
    assert s["commerce_returns"] == 60
    assert s["commerce_refunds"] == 40


# ---------------------------------------------------------------------------
# Seeded counts
# ---------------------------------------------------------------------------


def test_seed_persists_expected_commerce_row_counts(seeded_engine) -> None:
    engine, summary = seeded_engine
    s = SCALES["small"]
    with Session(engine) as session:
        n = lambda Tbl: session.execute(  # noqa: E731
            select(func.count()).select_from(Tbl)
        ).scalar_one()
        assert n(ProductCategory) == s["product_categories"]
        assert n(Product) == s["products"]
        assert n(ProductAttribute) == s["product_attributes"]
        assert n(ProductPrice) == s["product_prices"]
        assert n(Warehouse) == s["warehouses"]
        assert n(ProductInventory) == s["product_inventory"]
        assert n(CommerceOrder) == s["commerce_orders"]
        assert n(CommerceOrderItem) == s["commerce_order_items"]
        assert n(Shipment) == s["shipments"]
        assert n(CommerceReturn) == s["commerce_returns"]
        assert n(CommerceRefund) == s["commerce_refunds"]
    for key in _COMMERCE_KEYS:
        assert summary[key] == s[key], key


def test_commerce_seed_did_not_disturb_airline_or_saas_counts(seeded_engine) -> None:
    _, summary = seeded_engine
    s = SCALES["small"]
    for key in (
        # airline + support
        "customers", "airports", "flights", "bookings", "seats",
        "baggage_rules", "refunds", "support_tickets", "support_messages",
        "kb_articles",
        # saas
        "organizations", "customer_organizations", "plans", "subscriptions",
        "invoices", "invoice_items", "usage_events", "api_usage_daily",
        "seat_allocations", "overage_charges",
    ):
        assert summary[key] == s[key], key


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def test_commerce_orders_link_real_customers(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        orphans = s.execute(
            select(func.count())
            .select_from(CommerceOrder)
            .outerjoin(Customer, CommerceOrder.customer_id == Customer.id)
            .where(Customer.id.is_(None))
        ).scalar_one()
    assert orphans == 0


def test_order_items_link_real_orders_and_products(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad_order = s.execute(
            select(func.count())
            .select_from(CommerceOrderItem)
            .outerjoin(CommerceOrder, CommerceOrderItem.order_id == CommerceOrder.id)
            .where(CommerceOrder.id.is_(None))
        ).scalar_one()
        bad_product = s.execute(
            select(func.count())
            .select_from(CommerceOrderItem)
            .outerjoin(Product, CommerceOrderItem.product_id == Product.id)
            .where(Product.id.is_(None))
        ).scalar_one()
    assert bad_order == 0
    assert bad_product == 0


def test_shipments_link_real_orders(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        orphans = s.execute(
            select(func.count())
            .select_from(Shipment)
            .outerjoin(CommerceOrder, Shipment.order_id == CommerceOrder.id)
            .where(CommerceOrder.id.is_(None))
        ).scalar_one()
    assert orphans == 0


def test_returns_link_real_orders(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        orphans = s.execute(
            select(func.count())
            .select_from(CommerceReturn)
            .outerjoin(CommerceOrder, CommerceReturn.order_id == CommerceOrder.id)
            .where(CommerceOrder.id.is_(None))
        ).scalar_one()
    assert orphans == 0


def test_commerce_refunds_link_real_returns(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        orphans = s.execute(
            select(func.count())
            .select_from(CommerceRefund)
            .outerjoin(CommerceReturn, CommerceRefund.return_id == CommerceReturn.id)
            .where(CommerceReturn.id.is_(None))
        ).scalar_one()
    assert orphans == 0


def test_inventory_links_real_products_and_warehouses(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad_product = s.execute(
            select(func.count())
            .select_from(ProductInventory)
            .outerjoin(Product, ProductInventory.product_id == Product.id)
            .where(Product.id.is_(None))
        ).scalar_one()
        bad_wh = s.execute(
            select(func.count())
            .select_from(ProductInventory)
            .outerjoin(Warehouse, ProductInventory.warehouse_id == Warehouse.id)
            .where(Warehouse.id.is_(None))
        ).scalar_one()
    assert bad_product == 0
    assert bad_wh == 0


# ---------------------------------------------------------------------------
# Unique constraints + structural invariants
# ---------------------------------------------------------------------------


def test_product_inventory_unique_per_warehouse_product(seeded_engine) -> None:
    engine, _ = seeded_engine
    Session_ = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    with Session_() as s:
        existing = s.execute(select(ProductInventory).limit(1)).scalar_one()
        s.add(
            ProductInventory(
                product_id=existing.product_id,
                warehouse_id=existing.warehouse_id,
                quantity_available=1,
            )
        )
        with pytest.raises(IntegrityError):
            s.flush()
        s.rollback()


def test_product_categories_have_hierarchy(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        top_level = s.execute(
            select(func.count()).select_from(ProductCategory).where(
                ProductCategory.parent_id.is_(None)
            )
        ).scalar_one()
        children = s.execute(
            select(func.count()).select_from(ProductCategory).where(
                ProductCategory.parent_id.is_not(None)
            )
        ).scalar_one()
    assert top_level >= 1
    assert children >= 1


def test_at_least_some_customers_have_orders(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        n_customers_with_orders = s.execute(
            select(func.count(func.distinct(CommerceOrder.customer_id)))
        ).scalar_one()
    # 300 orders across 500 customers → ~ 200+ distinct customers
    assert n_customers_with_orders >= 50


def test_order_has_items_and_shipment(seeded_engine) -> None:
    """Pick a shipped order and confirm it has items + a shipment."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        order = s.execute(
            select(CommerceOrder)
            .where(CommerceOrder.status.in_(("shipped", "delivered")))
            .limit(1)
        ).scalar_one_or_none()
        assert order is not None, "expected at least one shipped/delivered order"
        n_items = s.execute(
            select(func.count())
            .select_from(CommerceOrderItem)
            .where(CommerceOrderItem.order_id == order.id)
        ).scalar_one()
        n_ship = s.execute(
            select(func.count())
            .select_from(Shipment)
            .where(Shipment.order_id == order.id)
        ).scalar_one()
        assert n_items >= 1
        assert n_ship >= 1


def test_refunds_only_on_approved_or_completed_returns(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad = s.execute(
            select(func.count())
            .select_from(CommerceRefund)
            .join(CommerceReturn, CommerceReturn.id == CommerceRefund.return_id)
            .where(~CommerceReturn.status.in_(("approved", "completed")))
        ).scalar_one()
    assert bad == 0


def test_current_price_exists_for_every_product(seeded_engine) -> None:
    """Each product has at least one price row with valid_to NULL."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        products_with_current = s.execute(
            select(func.count(func.distinct(ProductPrice.product_id)))
            .where(ProductPrice.valid_to.is_(None))
        ).scalar_one()
        total_products = s.execute(
            select(func.count()).select_from(Product)
        ).scalar_one()
    assert products_with_current == total_products


def test_product_attributes_link_real_products(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad = s.execute(
            select(func.count())
            .select_from(ProductAttribute)
            .outerjoin(Product, ProductAttribute.product_id == Product.id)
            .where(Product.id.is_(None))
        ).scalar_one()
    assert bad == 0
