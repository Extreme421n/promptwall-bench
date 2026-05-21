"""End-to-end tests for the 8 Phase 1D tools.

Each tool gets at least one happy-path test, one missing/invalid-input test,
and a not-found test where applicable. The registry-level tests also exercise
``invoke_tool`` (the trace-ready entry point).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Booking, Customer, Flight, KBArticle, SupportTicket
from app.tools import (
    AmbiguousInputError,
    InvocationResult,
    ResourceNotFoundError,
    ToolNotFoundError,
    ToolValidationError,
    default_registry,
    get_baggage_policy,
    get_booking_details,
    get_customer_profile,
    get_flight_status,
    get_refund_status,
    get_support_ticket_status,
    invoke_tool,
    search_available_flights,
    search_kb_articles,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_ALL_REGISTERED_TOOLS = sorted(
    [
        # Phase 1D
        "get_customer_profile",
        "get_booking_details",
        "get_flight_status",
        "search_available_flights",
        "get_refund_status",
        "get_baggage_policy",
        "get_support_ticket_status",
        "search_kb_articles",
        # Phase 2E
        "search_available_seats",
        "calculate_change_fee",
        "search_change_options",
        "get_loyalty_balance",
        "get_policy_clause",
        "get_customer_open_issues",
        "search_customer_records",
        # Phase C1 — SaaS / billing
        "get_subscription_status",
        "get_plan_limits",
        "get_invoice_status",
        "calculate_usage_overage",
        "get_api_usage_summary",
        "get_saas_seat_allocation",
        # Phase C1 — Commerce
        "search_products",
        "get_product_details",
        "check_product_inventory",
        "get_commerce_order_status",
        "get_commerce_refund_status",
        "get_shipment_status",
        # Phase C2 — Support extras
        "search_support_tickets",
        "get_escalation_policy",
        "create_support_ticket_draft",
        # Phase C2 — KB / policy extras
        "search_policy_documents",
        "get_latest_policy_version",
        # Phase C2 — Commerce extras
        "calculate_bundle_price",
        "get_commerce_return_status",
        # Phase C2 — CRM extras
        "get_customer_segment",
        # Phase 6B-4 — textual retrieval
        "search_return_rules",
        "get_product_warranty_terms",
        "search_internal_agent_notes",
        "search_operational_incidents",
        "get_support_resolution_template",
        "list_policy_versions",
        "get_active_policy",
    ]
)


def test_registry_has_all_registered_tools() -> None:
    names = default_registry.names()
    assert names == _ALL_REGISTERED_TOOLS
    assert len(default_registry) == len(_ALL_REGISTERED_TOOLS) == 42


def test_registry_describe_all_yields_function_calling_spec() -> None:
    descs = default_registry.describe_all()
    assert len(descs) == 42
    for d in descs:
        assert {"name", "description", "domain", "risk_level", "read_only",
                "input_schema", "output_schema"} <= set(d)
        assert d["input_schema"]["type"] == "object"
        assert d["output_schema"]["type"] == "object"


def test_registry_get_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotFoundError):
        default_registry.get("does_not_exist")


def test_invoke_tool_returns_invocation_result(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    result = invoke_tool("get_customer_profile", {"customer_id": cust_id}, seeded_session)
    assert isinstance(result, InvocationResult)
    assert result.success is True
    assert result.output is not None
    assert result.output["customer_id"] == cust_id
    assert result.error_type is None
    assert result.latency_ms >= 0


def test_invoke_tool_captures_validation_error(seeded_session: Session) -> None:
    result = invoke_tool("get_customer_profile", {}, seeded_session)
    assert result.success is False
    assert result.error_type == "ToolValidationError"
    assert "exactly one" in (result.error_message or "")


def test_invoke_tool_handles_unknown_tool(seeded_session: Session) -> None:
    result = invoke_tool("not_a_tool", {}, seeded_session)
    assert result.success is False
    assert result.error_type == "ToolNotFoundError"


# ---------------------------------------------------------------------------
# get_customer_profile
# ---------------------------------------------------------------------------


def test_get_customer_profile_by_customer_id(seeded_session: Session) -> None:
    cust = seeded_session.execute(select(Customer).limit(1)).scalar_one()
    out = get_customer_profile.call(seeded_session, {"customer_id": cust.id})
    assert out["customer_id"] == cust.id
    assert out["external_customer_id"] == cust.external_customer_id
    assert out["email"] == cust.email
    assert isinstance(out["has_loyalty"], bool)


def test_get_customer_profile_by_external_id(seeded_session: Session) -> None:
    out = get_customer_profile.call(
        seeded_session, {"external_customer_id": "CUST-00001"}
    )
    assert out["external_customer_id"] == "CUST-00001"


def test_get_customer_profile_by_email(seeded_session: Session) -> None:
    email = seeded_session.execute(select(Customer.email).limit(1)).scalar_one()
    out = get_customer_profile.call(seeded_session, {"email": email})
    assert out["email"] == email


def test_get_customer_profile_missing_input_raises(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_customer_profile.call(seeded_session, {})


def test_get_customer_profile_too_many_inputs_raises(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_customer_profile.call(
            seeded_session, {"customer_id": 1, "email": "x@y.com"}
        )


def test_get_customer_profile_not_found_raises(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_customer_profile.call(seeded_session, {"customer_id": 10_000_000})


# ---------------------------------------------------------------------------
# get_booking_details
# ---------------------------------------------------------------------------


def test_get_booking_details_by_reference(seeded_session: Session) -> None:
    ref = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    out = get_booking_details.call(seeded_session, {"booking_reference": ref})
    assert out["count"] == 1
    b = out["bookings"][0]
    assert b["booking_reference"] == ref
    assert b["flight_number"]
    assert b["booking_status"]
    assert b["cabin_class"]


def test_get_booking_details_by_customer_id(seeded_session: Session) -> None:
    # Find a customer with bookings
    cust_id = seeded_session.execute(
        select(Booking.customer_id).limit(1)
    ).scalar_one()
    out = get_booking_details.call(seeded_session, {"customer_id": cust_id})
    assert out["count"] >= 1
    assert all(b["customer_id"] == cust_id for b in out["bookings"])


def test_get_booking_details_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_booking_details.call(seeded_session, {})


def test_get_booking_details_unknown_reference(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_booking_details.call(seeded_session, {"booking_reference": "NOPE99"})


# ---------------------------------------------------------------------------
# get_flight_status
# ---------------------------------------------------------------------------


def test_get_flight_status_by_booking_reference(seeded_session: Session) -> None:
    ref = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    out = get_flight_status.call(seeded_session, {"booking_reference": ref})
    assert out["count"] == 1
    f = out["flights"][0]
    assert f["flight_number"]
    assert len(f["origin_code"]) >= 3
    assert len(f["destination_code"]) >= 3


def test_get_flight_status_by_flight_number(seeded_session: Session) -> None:
    fnum = seeded_session.execute(select(Flight.flight_number).limit(1)).scalar_one()
    out = get_flight_status.call(seeded_session, {"flight_number": fnum})
    assert out["count"] >= 1
    assert all(f["flight_number"] == fnum for f in out["flights"])


def test_get_flight_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_flight_status.call(seeded_session, {})


def test_get_flight_status_unknown_flight(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_flight_status.call(seeded_session, {"flight_number": "XX9999"})


# ---------------------------------------------------------------------------
# search_available_flights
# ---------------------------------------------------------------------------


def test_search_available_flights_happy(seeded_session: Session) -> None:
    # Pick a flight in the future to ensure date_from/date_to brackets it.
    now = datetime.now(timezone.utc)
    future_flight = seeded_session.execute(
        select(Flight)
        .where(Flight.scheduled_departure > now, Flight.status == "scheduled")
        .limit(1)
    ).scalar_one_or_none()
    if future_flight is None:
        pytest.skip("no future scheduled flights in seed (unexpected for small preset)")

    origin_code = seeded_session.execute(
        select(Flight.origin_airport_id).where(Flight.id == future_flight.id)
    ).scalar_one()
    from app.models import Airport
    origin = seeded_session.get(Airport, origin_code).code
    dest = seeded_session.get(Airport, future_flight.destination_airport_id).code

    dep_date = future_flight.scheduled_departure.date()
    out = search_available_flights.call(
        seeded_session,
        {
            "origin": origin,
            "destination": dest,
            "date_from": dep_date.isoformat(),
            "date_to": (dep_date + timedelta(days=1)).isoformat(),
        },
    )
    assert out["count"] >= 1
    flight_numbers = {f["flight_number"] for f in out["flights"]}
    assert future_flight.flight_number in flight_numbers


def test_search_available_flights_unknown_airport(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        search_available_flights.call(
            seeded_session,
            {
                "origin": "ZZZ",
                "destination": "JFK",
                "date_from": "2026-01-01",
                "date_to": "2026-12-31",
            },
        )


def test_search_available_flights_missing_required(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_available_flights.call(
            seeded_session, {"origin": "JFK", "destination": "LHR"}
        )


def test_search_available_flights_origin_equals_destination(
    seeded_session: Session,
) -> None:
    with pytest.raises(ToolValidationError):
        search_available_flights.call(
            seeded_session,
            {
                "origin": "JFK",
                "destination": "JFK",
                "date_from": "2026-01-01",
                "date_to": "2026-12-31",
            },
        )


# ---------------------------------------------------------------------------
# get_refund_status
# ---------------------------------------------------------------------------


def test_get_refund_status_by_booking_reference(seeded_session: Session) -> None:
    from app.models import Refund
    # Find a booking that has a refund.
    ref = seeded_session.execute(
        select(Booking.booking_reference)
        .join(Refund, Refund.booking_id == Booking.id)
        .limit(1)
    ).scalar_one()
    out = get_refund_status.call(seeded_session, {"booking_reference": ref})
    assert out["count"] >= 1
    assert out["refunds"][0]["booking_reference"] == ref


def test_get_refund_status_by_customer_id_with_no_refunds(seeded_session: Session) -> None:
    # Most customers have no refunds; pick one that exists but has none.
    from app.models import Refund
    cust_with_refund_ids = {
        r for (r,) in seeded_session.execute(
            select(Booking.customer_id).join(Refund, Refund.booking_id == Booking.id)
        ).all()
    }
    all_cust_ids = [c for (c,) in seeded_session.execute(select(Customer.id)).all()]
    candidates = [c for c in all_cust_ids if c not in cust_with_refund_ids]
    out = get_refund_status.call(seeded_session, {"customer_id": candidates[0]})
    assert out["count"] == 0
    assert out["refunds"] == []


def test_get_refund_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_refund_status.call(seeded_session, {})


def test_get_refund_status_unknown_booking(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_refund_status.call(seeded_session, {"booking_reference": "NOPE99"})


# ---------------------------------------------------------------------------
# get_baggage_policy
# ---------------------------------------------------------------------------


def test_get_baggage_policy_specific_route(seeded_session: Session) -> None:
    out = get_baggage_policy.call(
        seeded_session,
        {"cabin_class": "economy", "route_type": "international"},
    )
    assert out["count"] == 1
    p = out["policies"][0]
    assert p["cabin_class"] == "economy"
    assert p["route_type"] == "international"
    assert p["checked_bag_kg"] >= 0


def test_get_baggage_policy_all_routes_for_cabin(seeded_session: Session) -> None:
    out = get_baggage_policy.call(seeded_session, {"cabin_class": "business"})
    routes = {p["route_type"] for p in out["policies"]}
    # All four route types present in seed
    assert routes == {"domestic", "intra-continental", "international", "ultra-long-haul"}


def test_get_baggage_policy_unknown_cabin(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_baggage_policy.call(seeded_session, {"cabin_class": "captain"})


def test_get_baggage_policy_missing_cabin(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_baggage_policy.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_support_ticket_status
# ---------------------------------------------------------------------------


def test_get_support_ticket_status_by_number(seeded_session: Session) -> None:
    number = seeded_session.execute(
        select(SupportTicket.ticket_number).limit(1)
    ).scalar_one()
    out = get_support_ticket_status.call(seeded_session, {"ticket_number": number})
    assert out["count"] == 1
    t = out["tickets"][0]
    assert t["ticket_number"] == number
    assert t["message_count"] >= 1
    assert t["last_message_excerpt"] is not None


def test_get_support_ticket_status_by_customer(seeded_session: Session) -> None:
    cust_id = seeded_session.execute(
        select(SupportTicket.customer_id).limit(1)
    ).scalar_one()
    out = get_support_ticket_status.call(seeded_session, {"customer_id": cust_id})
    assert out["count"] >= 1
    assert all(t["customer_id"] == cust_id for t in out["tickets"])


def test_get_support_ticket_status_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_support_ticket_status.call(seeded_session, {})


def test_get_support_ticket_status_not_found(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_support_ticket_status.call(
            seeded_session, {"ticket_number": "TKT-NOPENO"}
        )


# ---------------------------------------------------------------------------
# search_kb_articles
# ---------------------------------------------------------------------------


def test_search_kb_articles_finds_baggage_results(seeded_session: Session) -> None:
    out = search_kb_articles.call(seeded_session, {"query": "baggage"})
    assert out["count"] >= 1
    assert all(
        "baggage" in (a["title"] + a["excerpt"]).lower() for a in out["articles"]
    )


def test_search_kb_articles_with_category_filter(seeded_session: Session) -> None:
    out = search_kb_articles.call(
        seeded_session, {"query": "refund", "category": "refunds"}
    )
    assert all(a["category"] == "refunds" for a in out["articles"])
    assert out["count"] >= 1


def test_search_kb_articles_no_results_returns_empty(seeded_session: Session) -> None:
    out = search_kb_articles.call(
        seeded_session, {"query": "zzzzzzzzzzzzzzz_no_match_xyzzy"}
    )
    assert out["count"] == 0
    assert out["articles"] == []


def test_search_kb_articles_missing_query(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_kb_articles.call(seeded_session, {})


def test_search_kb_articles_query_too_short(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_kb_articles.call(seeded_session, {"query": "a"})
