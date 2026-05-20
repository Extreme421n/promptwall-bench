"""Tests for the 8 tools added in Phase C2."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    CommerceOrder,
    CommerceReturn,
    Customer,
    Product,
    SupportTicket,
)
from app.tools import (
    ResourceNotFoundError,
    ToolValidationError,
    calculate_bundle_price,
    create_support_ticket_draft,
    get_commerce_return_status,
    get_customer_segment,
    get_escalation_policy,
    get_latest_policy_version,
    search_policy_documents,
    search_support_tickets,
)


# ---------------------------------------------------------------------------
# search_support_tickets
# ---------------------------------------------------------------------------


def test_search_support_tickets_returns_matches(seeded_session: Session) -> None:
    out = search_support_tickets.call(seeded_session, {"query": "refund"})
    assert out["count"] >= 1
    for t in out["tickets"]:
        assert "refund" in t["subject"].lower()


def test_search_support_tickets_status_filter(seeded_session: Session) -> None:
    out = search_support_tickets.call(
        seeded_session, {"query": "refund", "status": "open"}
    )
    assert all(t["status"] == "open" for t in out["tickets"])


def test_search_support_tickets_customer_filter(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(
        select(SupportTicket.customer_id).limit(1)
    ).scalar_one()
    out = search_support_tickets.call(
        seeded_session,
        {"query": "refund", "customer_id": cust_id},
    )
    assert all(t["customer_id"] == cust_id for t in out["tickets"])


def test_search_support_tickets_invalid_status_raises(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_support_tickets.call(
            seeded_session, {"query": "refund", "status": "snoozed"}
        )


def test_search_support_tickets_query_too_short(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_support_tickets.call(seeded_session, {"query": "a"})


def test_search_support_tickets_no_match_returns_empty(seeded_session: Session) -> None:
    out = search_support_tickets.call(
        seeded_session, {"query": "qqzqzxnomatchxxx"}
    )
    assert out["count"] == 0
    assert out["tickets"] == []


# ---------------------------------------------------------------------------
# get_escalation_policy
# ---------------------------------------------------------------------------


def test_escalation_policy_by_priority(seeded_session: Session) -> None:
    out = get_escalation_policy.call(seeded_session, {"priority": "high"})
    assert out["priority"] == "high"
    assert out["resolved_via"] == "priority"
    assert out["first_response_sla_hours"] >= 1
    assert out["resolution_sla_hours"] >= 1
    assert len(out["steps"]) >= 1


def test_escalation_policy_by_ticket_number(seeded_session: Session) -> None:
    tn = seeded_session.execute(select(SupportTicket.ticket_number).limit(1)).scalar_one()
    out = get_escalation_policy.call(seeded_session, {"ticket_number": tn})
    assert out["ticket_number"] == tn
    assert out["resolved_via"] == "ticket_number"
    assert out["priority"] in {"low", "normal", "high", "urgent"}


def test_escalation_policy_unknown_ticket_raises(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_escalation_policy.call(seeded_session, {"ticket_number": "TKT-NOPE99"})


def test_escalation_policy_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_escalation_policy.call(seeded_session, {})


def test_escalation_policy_invalid_priority(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_escalation_policy.call(seeded_session, {"priority": "extreme"})


# ---------------------------------------------------------------------------
# create_support_ticket_draft
# ---------------------------------------------------------------------------


def test_create_support_ticket_draft_returns_draft(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    out = create_support_ticket_draft.call(
        seeded_session,
        {
            "customer_id": cust_id,
            "subject": "Refund taking too long for booking",
            "priority": "high",
            "description": "We've been waiting 14 days for the refund.",
        },
    )
    assert out["is_draft"] is True
    assert out["customer_id"] == cust_id
    assert out["draft_ticket"]["proposed_status"] == "draft_pending_review"
    assert "NOT been persisted" in out["next_steps"]


def test_create_support_ticket_draft_does_not_persist(seeded_session: Session) -> None:
    """The DB row count must NOT grow after a draft call."""
    from sqlalchemy import func

    before = seeded_session.execute(
        select(func.count()).select_from(SupportTicket)
    ).scalar_one()
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    create_support_ticket_draft.call(
        seeded_session,
        {"customer_id": cust_id, "subject": "Lost luggage at JFK"},
    )
    after = seeded_session.execute(
        select(func.count()).select_from(SupportTicket)
    ).scalar_one()
    assert after == before


def test_create_support_ticket_draft_unknown_customer(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        create_support_ticket_draft.call(
            seeded_session,
            {"customer_id": 99_999_999, "subject": "Generic subject"},
        )


def test_create_support_ticket_draft_invalid_priority(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    with pytest.raises(ToolValidationError):
        create_support_ticket_draft.call(
            seeded_session,
            {
                "customer_id": cust_id,
                "subject": "Some subject here",
                "priority": "extreme",
            },
        )


def test_create_support_ticket_draft_subject_too_short(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    with pytest.raises(ToolValidationError):
        create_support_ticket_draft.call(
            seeded_session, {"customer_id": cust_id, "subject": "hi"}
        )


# ---------------------------------------------------------------------------
# search_policy_documents
# ---------------------------------------------------------------------------


def test_search_policy_documents_finds_baggage(seeded_session: Session) -> None:
    out = search_policy_documents.call(seeded_session, {"query": "baggage"})
    assert out["count"] >= 1
    assert all(d["category"] in {
        "baggage", "refunds", "flight_change",
        "cancellation", "loyalty", "special_assistance",
    } for d in out["documents"])


def test_search_policy_documents_with_category(seeded_session: Session) -> None:
    out = search_policy_documents.call(
        seeded_session, {"query": "refund", "category": "refunds"}
    )
    assert all(d["category"] == "refunds" for d in out["documents"])


def test_search_policy_documents_non_policy_category_returns_empty(
    seeded_session: Session,
) -> None:
    """`check_in` exists in the KB but is not a policy category — so this
    search must return no rows, not 404."""
    out = search_policy_documents.call(
        seeded_session, {"query": "check", "category": "check_in"}
    )
    assert out["count"] == 0


def test_search_policy_documents_missing_query(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_policy_documents.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_latest_policy_version
# ---------------------------------------------------------------------------


def test_get_latest_policy_version_happy_path(seeded_session: Session) -> None:
    out = get_latest_policy_version.call(seeded_session, {"slug": "refunds-5"})
    assert out["slug"] == "refunds-5"
    assert out["version"] >= 1
    assert out["body_excerpt"]


def test_get_latest_policy_version_unknown(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_latest_policy_version.call(seeded_session, {"slug": "nope-99"})


def test_get_latest_policy_version_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_latest_policy_version.call(seeded_session, {})


# ---------------------------------------------------------------------------
# calculate_bundle_price
# ---------------------------------------------------------------------------


def test_calculate_bundle_price_basic(seeded_session: Session) -> None:
    skus = list(
        seeded_session.execute(select(Product.sku).limit(3)).scalars().all()
    )
    out = calculate_bundle_price.call(
        seeded_session,
        {
            "items": [
                {"sku": skus[0], "quantity": 2},
                {"sku": skus[1], "quantity": 1},
                {"sku": skus[2], "quantity": 5},
            ]
        },
    )
    assert out["item_count"] == 8
    assert float(out["subtotal"]) > 0
    # No discount → total == subtotal
    assert float(out["total"]) == float(out["subtotal"])
    assert len(out["items"]) == 3


def test_calculate_bundle_price_with_discount(seeded_session: Session) -> None:
    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    out = calculate_bundle_price.call(
        seeded_session,
        {"items": [{"sku": sku, "quantity": 1}], "discount_pct": 10.0},
    )
    assert float(out["discount_amount"]) > 0
    assert float(out["total"]) < float(out["subtotal"])


def test_calculate_bundle_price_unknown_sku(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        calculate_bundle_price.call(
            seeded_session,
            {"items": [{"sku": "SKU-NOPE-99", "quantity": 1}]},
        )


def test_calculate_bundle_price_empty_items(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        calculate_bundle_price.call(seeded_session, {"items": []})


def test_calculate_bundle_price_invalid_discount(seeded_session: Session) -> None:
    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    with pytest.raises(ToolValidationError):
        calculate_bundle_price.call(
            seeded_session,
            {"items": [{"sku": sku, "quantity": 1}], "discount_pct": 99.0},
        )


# ---------------------------------------------------------------------------
# get_commerce_return_status
# ---------------------------------------------------------------------------


def test_commerce_return_status_by_order(seeded_session: Session) -> None:
    on = seeded_session.execute(
        select(CommerceOrder.order_number)
        .join(CommerceReturn, CommerceReturn.order_id == CommerceOrder.id)
        .limit(1)
    ).scalar_one()
    out = get_commerce_return_status.call(seeded_session, {"order_number": on})
    assert out["count"] >= 1
    assert all(r["order_number"] == on for r in out["returns"])


def test_commerce_return_status_by_customer(seeded_session: Session) -> None:
    cust = seeded_session.execute(
        select(CommerceOrder.customer_id).limit(1)
    ).scalar_one()
    out = get_commerce_return_status.call(seeded_session, {"customer_id": cust})
    assert "returns" in out


def test_commerce_return_status_unknown_order(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_commerce_return_status.call(
            seeded_session, {"order_number": "ORD-NOPE99"}
        )


def test_commerce_return_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_commerce_return_status.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_customer_segment
# ---------------------------------------------------------------------------


def test_customer_segment_by_customer_id(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    out = get_customer_segment.call(seeded_session, {"customer_id": cust_id})
    assert out["customer_id"] == cust_id
    assert out["external_customer_id"].startswith("CUST-")
    assert out["booking_count"] >= 0
    assert out["commerce_order_count"] >= 0
    assert out["organization_count"] >= 0
    assert out["support_ticket_count"] >= 0


def test_customer_segment_by_external_id(seeded_session: Session) -> None:
    out = get_customer_segment.call(
        seeded_session, {"external_customer_id": "CUST-00001"}
    )
    assert out["external_customer_id"] == "CUST-00001"


def test_customer_segment_by_email(seeded_session: Session) -> None:
    email = seeded_session.execute(select(Customer.email).limit(1)).scalar_one()
    out = get_customer_segment.call(seeded_session, {"email": email})
    assert out["customer_id"]


def test_customer_segment_not_found(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_customer_segment.call(seeded_session, {"customer_id": 99_999_999})


def test_customer_segment_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_customer_segment.call(seeded_session, {})
