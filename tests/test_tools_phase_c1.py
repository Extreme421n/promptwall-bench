"""Tests for the 12 SaaS + Commerce tools added in Phase C1."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    CommerceOrder,
    CommerceRefund,
    CommerceReturn,
    CustomerOrganization,
    Invoice,
    Organization,
    Plan,
    Product,
    SeatAllocation,
    Shipment,
    Subscription,
)
from app.tools import (
    ResourceNotFoundError,
    ToolValidationError,
    calculate_usage_overage,
    check_product_inventory,
    get_api_usage_summary,
    get_commerce_order_status,
    get_commerce_refund_status,
    get_invoice_status,
    get_plan_limits,
    get_product_details,
    get_saas_seat_allocation,
    get_shipment_status,
    get_subscription_status,
    search_products,
)


# ---------------------------------------------------------------------------
# get_subscription_status
# ---------------------------------------------------------------------------


def test_subscription_status_by_organization_id(seeded_session: Session) -> None:
    org_id = seeded_session.execute(select(Subscription.organization_id).limit(1)).scalar_one()
    out = get_subscription_status.call(seeded_session, {"organization_id": org_id})
    assert out["count"] >= 1
    sub = out["subscriptions"][0]
    assert sub["organization_id"] == org_id
    assert sub["plan_name"]
    assert sub["plan_tier"] in ("starter", "pro", "business", "enterprise")


def test_subscription_status_by_customer_id(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(
        select(CustomerOrganization.customer_id).limit(1)
    ).scalar_one()
    out = get_subscription_status.call(seeded_session, {"customer_id": cust_id})
    # Customer may be in 0+ orgs that have subscriptions; just check schema.
    assert "subscriptions" in out


def test_subscription_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_subscription_status.call(seeded_session, {})


def test_subscription_status_unknown_org(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_subscription_status.call(seeded_session, {"organization_id": 9_999_999})


# ---------------------------------------------------------------------------
# get_plan_limits
# ---------------------------------------------------------------------------


def test_plan_limits_by_plan_name(seeded_session: Session) -> None:
    out = get_plan_limits.call(seeded_session, {"plan_name": "Pro"})
    assert out["plan_name"] == "Pro"
    assert out["tier"] == "pro"
    assert out["included_seats"] >= 1
    assert out["included_api_calls"] >= 1
    assert out["resolved_via"] == "plan_name"


def test_plan_limits_by_organization_id(seeded_session: Session) -> None:
    org_id = seeded_session.execute(select(Subscription.organization_id).limit(1)).scalar_one()
    out = get_plan_limits.call(seeded_session, {"organization_id": org_id})
    assert out["plan_name"]
    assert out["resolved_via"] == "organization_id"


def test_plan_limits_unknown_plan(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_plan_limits.call(seeded_session, {"plan_name": "NoSuchPlan"})


def test_plan_limits_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_plan_limits.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_invoice_status
# ---------------------------------------------------------------------------


def test_invoice_status_by_invoice_number(seeded_session: Session) -> None:
    number = seeded_session.execute(select(Invoice.invoice_number).limit(1)).scalar_one()
    out = get_invoice_status.call(seeded_session, {"invoice_number": number})
    assert out["count"] == 1
    inv = out["invoices"][0]
    assert inv["invoice_number"] == number
    assert inv["status"]


def test_invoice_status_by_organization(seeded_session: Session) -> None:
    org_id = seeded_session.execute(select(Invoice.organization_id).limit(1)).scalar_one()
    out = get_invoice_status.call(seeded_session, {"organization_id": org_id})
    assert out["count"] >= 1
    assert all(i["organization_id"] == org_id for i in out["invoices"])


def test_invoice_status_unknown_invoice(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_invoice_status.call(seeded_session, {"invoice_number": "INV-NOPE99"})


def test_invoice_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_invoice_status.call(seeded_session, {})


# ---------------------------------------------------------------------------
# calculate_usage_overage
# ---------------------------------------------------------------------------


def test_calculate_usage_overage_happy_path(seeded_session: Session) -> None:
    org_id = seeded_session.execute(select(Organization.id).limit(1)).scalar_one()
    today = date.today()
    out = calculate_usage_overage.call(
        seeded_session,
        {
            "organization_id": org_id,
            "date_from": (today - timedelta(days=30)).isoformat(),
            "date_to": today.isoformat(),
        },
    )
    assert out["organization_id"] == org_id
    assert out["days_in_range"] == 31
    assert out["total_api_calls"] >= 0
    assert out["included_quota"] >= 0
    assert out["overage_calls"] >= 0
    assert float(out["estimated_overage_charge_usd"]) >= 0.0


def test_calculate_usage_overage_invalid_range(seeded_session: Session) -> None:
    org_id = seeded_session.execute(select(Organization.id).limit(1)).scalar_one()
    with pytest.raises(ToolValidationError):
        calculate_usage_overage.call(
            seeded_session,
            {
                "organization_id": org_id,
                "date_from": "2026-12-01",
                "date_to": "2026-01-01",
            },
        )


def test_calculate_usage_overage_unknown_org(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        calculate_usage_overage.call(
            seeded_session,
            {
                "organization_id": 9_999_999,
                "date_from": "2026-01-01",
                "date_to": "2026-01-31",
            },
        )


# ---------------------------------------------------------------------------
# get_api_usage_summary
# ---------------------------------------------------------------------------


def test_api_usage_summary_happy_path(seeded_session: Session) -> None:
    org_id = seeded_session.execute(select(Organization.id).limit(1)).scalar_one()
    today = date.today()
    out = get_api_usage_summary.call(
        seeded_session,
        {
            "organization_id": org_id,
            "date_from": (today - timedelta(days=30)).isoformat(),
            "date_to": today.isoformat(),
        },
    )
    assert 0.0 <= out["success_rate"] <= 1.0
    assert out["total_calls"] == out["successful_calls"] + out["failed_calls"]


def test_api_usage_summary_missing_org(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_api_usage_summary.call(
            seeded_session, {"date_from": "2026-01-01", "date_to": "2026-01-31"}
        )


def test_api_usage_summary_unknown_org(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_api_usage_summary.call(
            seeded_session,
            {
                "organization_id": 9_999_999,
                "date_from": "2026-01-01",
                "date_to": "2026-01-31",
            },
        )


# ---------------------------------------------------------------------------
# get_saas_seat_allocation
# ---------------------------------------------------------------------------


def test_seat_allocation_happy_path(seeded_session: Session) -> None:
    org_id = seeded_session.execute(select(SeatAllocation.organization_id).limit(1)).scalar_one()
    out = get_saas_seat_allocation.call(seeded_session, {"organization_id": org_id})
    assert out["organization_id"] == org_id
    assert out["organization_name"]
    assert out["allocated_seats"] >= 0
    assert out["used_seats"] >= 0
    assert out["remaining_seats"] == max(0, out["allocated_seats"] - out["used_seats"])


def test_seat_allocation_unknown_org(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_saas_seat_allocation.call(seeded_session, {"organization_id": 9_999_999})


def test_seat_allocation_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_saas_seat_allocation.call(seeded_session, {})


# ---------------------------------------------------------------------------
# search_products
# ---------------------------------------------------------------------------


def test_search_products_returns_matches(seeded_session: Session) -> None:
    # Pick a real product noun and search for it.
    sample = seeded_session.execute(select(Product.name).limit(1)).scalar_one()
    fragment = sample.split()[0]  # e.g. "Wireless"
    out = search_products.call(seeded_session, {"query": fragment})
    assert out["count"] >= 1
    for p in out["products"]:
        assert fragment.lower() in p["name"].lower()


def test_search_products_max_price_filter(seeded_session: Session) -> None:
    out = search_products.call(
        seeded_session, {"query": "Headphones", "max_price": "1.00"}
    )
    # No product priced under $1 → empty result, not an error.
    assert out["count"] == 0
    assert out["products"] == []


def test_search_products_missing_query(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_products.call(seeded_session, {})


def test_search_products_query_too_short(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_products.call(seeded_session, {"query": "a"})


# ---------------------------------------------------------------------------
# get_product_details
# ---------------------------------------------------------------------------


def test_product_details_by_sku(seeded_session: Session) -> None:
    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    out = get_product_details.call(seeded_session, {"sku": sku})
    assert out["sku"] == sku
    assert out["name"]
    assert out["category_name"]
    assert isinstance(out["attributes"], list)
    assert out["total_inventory"] >= 0


def test_product_details_by_id(seeded_session: Session) -> None:
    pid = seeded_session.execute(select(Product.id).limit(1)).scalar_one()
    out = get_product_details.call(seeded_session, {"product_id": pid})
    assert out["product_id"] == pid


def test_product_details_unknown_sku(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_product_details.call(seeded_session, {"sku": "SKU-NOPE-99"})


def test_product_details_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_product_details.call(seeded_session, {})


# ---------------------------------------------------------------------------
# check_product_inventory
# ---------------------------------------------------------------------------


def test_check_product_inventory_by_sku(seeded_session: Session) -> None:
    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    out = check_product_inventory.call(seeded_session, {"sku": sku})
    assert out["sku"] == sku
    assert out["warehouse_count"] == len(out["inventory"])
    assert out["total_quantity"] == sum(i["quantity_available"] for i in out["inventory"])


def test_check_product_inventory_with_city_filter(seeded_session: Session) -> None:
    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    out = check_product_inventory.call(
        seeded_session, {"sku": sku, "city": "Reno"}
    )
    assert all("Reno" in i["city"] or i["city"] == "Reno" for i in out["inventory"])


def test_check_product_inventory_unknown(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        check_product_inventory.call(seeded_session, {"sku": "SKU-NOPE-99"})


def test_check_product_inventory_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        check_product_inventory.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_commerce_order_status
# ---------------------------------------------------------------------------


def test_commerce_order_status_by_order_number(seeded_session: Session) -> None:
    on = seeded_session.execute(select(CommerceOrder.order_number).limit(1)).scalar_one()
    out = get_commerce_order_status.call(seeded_session, {"order_number": on})
    assert out["count"] == 1
    o = out["orders"][0]
    assert o["order_number"] == on
    assert o["item_count"] >= 1


def test_commerce_order_status_by_customer(seeded_session: Session) -> None:
    cust = seeded_session.execute(select(CommerceOrder.customer_id).limit(1)).scalar_one()
    out = get_commerce_order_status.call(seeded_session, {"customer_id": cust})
    assert out["count"] >= 1
    assert all(o["customer_id"] == cust for o in out["orders"])


def test_commerce_order_status_unknown_order(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_commerce_order_status.call(seeded_session, {"order_number": "ORD-NOPE99"})


def test_commerce_order_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_commerce_order_status.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_commerce_refund_status
# ---------------------------------------------------------------------------


def test_commerce_refund_status_by_order(seeded_session: Session) -> None:
    on = seeded_session.execute(
        select(CommerceOrder.order_number)
        .join(CommerceReturn, CommerceReturn.order_id == CommerceOrder.id)
        .limit(1)
    ).scalar_one()
    out = get_commerce_refund_status.call(seeded_session, {"order_number": on})
    assert out["count"] >= 1


def test_commerce_refund_status_by_customer(seeded_session: Session) -> None:
    cust = seeded_session.execute(select(CommerceOrder.customer_id).limit(1)).scalar_one()
    out = get_commerce_refund_status.call(seeded_session, {"customer_id": cust})
    assert "refunds" in out


def test_commerce_refund_status_unknown_order(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_commerce_refund_status.call(
            seeded_session, {"order_number": "ORD-NOPE99"}
        )


def test_commerce_refund_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_commerce_refund_status.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_shipment_status
# ---------------------------------------------------------------------------


def test_shipment_status_by_tracking(seeded_session: Session) -> None:
    tn = seeded_session.execute(select(Shipment.tracking_number).limit(1)).scalar_one()
    out = get_shipment_status.call(seeded_session, {"tracking_number": tn})
    assert out["count"] == 1
    s = out["shipments"][0]
    assert s["tracking_number"] == tn
    assert s["carrier"]


def test_shipment_status_by_order(seeded_session: Session) -> None:
    # Find an order that has a shipment.
    on = seeded_session.execute(
        select(CommerceOrder.order_number)
        .join(Shipment, Shipment.order_id == CommerceOrder.id)
        .limit(1)
    ).scalar_one()
    out = get_shipment_status.call(seeded_session, {"order_number": on})
    assert out["count"] >= 1
    assert all(s["order_number"] == on for s in out["shipments"])


def test_shipment_status_unknown(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_shipment_status.call(seeded_session, {"tracking_number": "TRK-NOSUCH"})


def test_shipment_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_shipment_status.call(seeded_session, {})
