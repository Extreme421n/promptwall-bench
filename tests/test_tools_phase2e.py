"""Tests for the 7 tools added in Phase 2E."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking, Customer, Flight, SupportTicket
from app.tools import (
    ResourceNotFoundError,
    ToolValidationError,
    calculate_change_fee,
    get_customer_open_issues,
    get_loyalty_balance,
    get_policy_clause,
    search_available_seats,
    search_change_options,
    search_customer_records,
)


# ---------------------------------------------------------------------------
# search_available_seats
# ---------------------------------------------------------------------------


def test_search_available_seats_by_booking(seeded_session: Session) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    out = search_available_seats.call(seeded_session, {"booking_reference": pnr})
    assert out["flight_id"]
    assert isinstance(out["seats"], list)
    assert out["cabin_filter"] is None
    assert out["count"] == len(out["seats"])


def test_search_available_seats_by_flight(seeded_session: Session) -> None:
    fn = seeded_session.execute(select(Flight.flight_number).limit(1)).scalar_one()
    out = search_available_seats.call(seeded_session, {"flight_number": fn})
    assert out["flight_number"] == fn


def test_search_available_seats_with_cabin(seeded_session: Session) -> None:
    fn = seeded_session.execute(select(Flight.flight_number).limit(1)).scalar_one()
    out = search_available_seats.call(
        seeded_session, {"flight_number": fn, "cabin_class": "economy"}
    )
    assert out["cabin_filter"] == "economy"
    assert all(s["cabin_class"] == "economy" for s in out["seats"])


def test_search_available_seats_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_available_seats.call(seeded_session, {})


def test_search_available_seats_unknown_flight(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        search_available_seats.call(seeded_session, {"flight_number": "XX9999"})


def test_search_available_seats_invalid_cabin(seeded_session: Session) -> None:
    fn = seeded_session.execute(select(Flight.flight_number).limit(1)).scalar_one()
    with pytest.raises(ToolValidationError):
        search_available_seats.call(
            seeded_session, {"flight_number": fn, "cabin_class": "captain"}
        )


# ---------------------------------------------------------------------------
# calculate_change_fee
# ---------------------------------------------------------------------------


def test_calculate_change_fee_basic(seeded_session: Session) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    out = calculate_change_fee.call(seeded_session, {"booking_reference": pnr})
    assert out["booking_reference"] == pnr
    assert out["cabin_class"]
    assert float(out["change_fee"]) >= 0
    assert float(out["total_change_cost"]) >= float(out["change_fee"])
    assert out["currency"]


def test_calculate_change_fee_with_new_flight(seeded_session: Session) -> None:
    # Pick a booking + a new flight number that exists.
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    new_fn = seeded_session.execute(
        select(Flight.flight_number).offset(5).limit(1)
    ).scalar_one()
    out = calculate_change_fee.call(
        seeded_session, {"booking_reference": pnr, "new_flight_number": new_fn}
    )
    # When new flight is provided, fare difference is reported (possibly 0.00).
    assert out["new_fare_difference"] is not None


def test_calculate_change_fee_with_new_date(seeded_session: Session) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    out = calculate_change_fee.call(
        seeded_session,
        {"booking_reference": pnr, "new_date": "2026-08-15"},
    )
    assert "date-only change" in out["notes"].lower()


def test_calculate_change_fee_not_found(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        calculate_change_fee.call(seeded_session, {"booking_reference": "NOPE99"})


def test_calculate_change_fee_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        calculate_change_fee.call(seeded_session, {})


# ---------------------------------------------------------------------------
# search_change_options
# ---------------------------------------------------------------------------


def test_search_change_options_basic(seeded_session: Session) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    today = date.today()
    out = search_change_options.call(
        seeded_session,
        {
            "booking_reference": pnr,
            "date_from": today.isoformat(),
            "date_to": (today + timedelta(days=60)).isoformat(),
        },
    )
    assert out["booking_reference"] == pnr
    assert out["origin_code"]
    assert out["destination_code"]
    assert out["cabin_class"]
    assert isinstance(out["options"], list)
    assert out["count"] == len(out["options"])


def test_search_change_options_invalid_range(seeded_session: Session) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    with pytest.raises(ToolValidationError):
        search_change_options.call(
            seeded_session,
            {
                "booking_reference": pnr,
                "date_from": "2026-12-01",
                "date_to": "2026-01-01",
            },
        )


def test_search_change_options_unknown_booking(seeded_session: Session) -> None:
    today = date.today()
    with pytest.raises(ResourceNotFoundError):
        search_change_options.call(
            seeded_session,
            {
                "booking_reference": "NOPE99",
                "date_from": today.isoformat(),
                "date_to": (today + timedelta(days=30)).isoformat(),
            },
        )


# ---------------------------------------------------------------------------
# get_loyalty_balance
# ---------------------------------------------------------------------------


def test_get_loyalty_balance_by_customer_id(seeded_session: Session) -> None:
    # Pick a customer that has loyalty.
    from app.models import LoyaltyAccount

    cust_id = seeded_session.execute(
        select(LoyaltyAccount.customer_id).limit(1)
    ).scalar_one()
    out = get_loyalty_balance.call(seeded_session, {"customer_id": cust_id})
    assert out["customer_id"] == cust_id
    assert out["has_loyalty"] is True
    assert out["tier"]
    assert out["points_balance"] is not None


def test_get_loyalty_balance_by_email(seeded_session: Session) -> None:
    email = seeded_session.execute(select(Customer.email).limit(1)).scalar_one()
    out = get_loyalty_balance.call(seeded_session, {"email": email})
    assert out["customer_id"]


def test_get_loyalty_balance_no_account(seeded_session: Session) -> None:
    # Find a customer that does NOT have a loyalty account.
    from app.models import LoyaltyAccount

    with_loyalty = {
        r for (r,) in seeded_session.execute(select(LoyaltyAccount.customer_id)).all()
    }
    all_ids = [c for (c,) in seeded_session.execute(select(Customer.id)).all()]
    no_loyalty = next(c for c in all_ids if c not in with_loyalty)
    out = get_loyalty_balance.call(seeded_session, {"customer_id": no_loyalty})
    assert out["has_loyalty"] is False
    assert out["tier"] is None


def test_get_loyalty_balance_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_loyalty_balance.call(seeded_session, {})


def test_get_loyalty_balance_not_found(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_loyalty_balance.call(seeded_session, {"customer_id": 99_999_999})


# ---------------------------------------------------------------------------
# get_policy_clause
# ---------------------------------------------------------------------------


def test_get_policy_clause_finds_refund_clauses(seeded_session: Session) -> None:
    out = get_policy_clause.call(seeded_session, {"policy_topic": "refund"})
    assert out["count"] >= 1
    assert all(
        "refund" in (c["title"] + c["excerpt"]).lower() for c in out["clauses"]
    )


def test_get_policy_clause_with_category(seeded_session: Session) -> None:
    out = get_policy_clause.call(
        seeded_session,
        {"policy_topic": "baggage", "category": "baggage"},
    )
    assert all(c["category"] == "baggage" for c in out["clauses"])
    assert out["count"] >= 1


def test_get_policy_clause_no_results(seeded_session: Session) -> None:
    out = get_policy_clause.call(
        seeded_session, {"policy_topic": "zzz_no_match_xyzzy"}
    )
    assert out["count"] == 0
    assert out["clauses"] == []


def test_get_policy_clause_missing_topic(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_policy_clause.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_customer_open_issues
# ---------------------------------------------------------------------------


def test_get_customer_open_issues_returns_open_set(seeded_session: Session) -> None:
    # Pick a customer that has at least one open/pending ticket.
    cust_id = seeded_session.execute(
        select(SupportTicket.customer_id)
        .where(SupportTicket.status.in_(("open", "pending")))
        .limit(1)
    ).scalar_one()

    out = get_customer_open_issues.call(seeded_session, {"customer_id": cust_id})
    assert out["customer_id"] == cust_id
    assert out["open_ticket_count"] >= 1
    assert out["pending_refund_count"] >= 0
    assert out["open_ticket_count"] + out["pending_refund_count"] == len(out["issues"])
    # All issue rows are typed correctly.
    for issue in out["issues"]:
        assert issue["type"] in ("ticket", "refund")
        assert issue["identifier"]
        assert issue["status"]


def test_get_customer_open_issues_unknown_customer(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_customer_open_issues.call(seeded_session, {"customer_id": 99_999_999})


def test_get_customer_open_issues_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_customer_open_issues.call(seeded_session, {})


# ---------------------------------------------------------------------------
# search_customer_records
# ---------------------------------------------------------------------------


def test_search_customer_records_by_email(seeded_session: Session) -> None:
    email = seeded_session.execute(select(Customer.email).limit(1)).scalar_one()
    out = search_customer_records.call(seeded_session, {"email": email})
    assert out["count"] >= 1
    assert any(m["email"] == email for m in out["matches"])


def test_search_customer_records_by_name_fragment(seeded_session: Session) -> None:
    full = seeded_session.execute(select(Customer.full_name).limit(1)).scalar_one()
    fragment = full.split()[0]  # first name
    out = search_customer_records.call(seeded_session, {"full_name": fragment})
    assert out["count"] >= 1


def test_search_customer_records_by_phone(seeded_session: Session) -> None:
    phone = seeded_session.execute(
        select(Customer.phone).where(Customer.phone.is_not(None)).limit(1)
    ).scalar_one()
    # Use a substring to be robust to formatting differences.
    fragment = phone[:5] if phone else ""
    out = search_customer_records.call(seeded_session, {"phone": fragment})
    assert out["count"] >= 1


def test_search_customer_records_no_input_raises(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_customer_records.call(seeded_session, {})


def test_search_customer_records_returns_empty_when_no_match(
    seeded_session: Session,
) -> None:
    out = search_customer_records.call(
        seeded_session, {"email": "zzz_no_match_xyzzy@example.invalid"}
    )
    assert out["count"] == 0
    assert out["matches"] == []
