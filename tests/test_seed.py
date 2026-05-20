"""End-to-end tests for the seed module.

Uses the shared ``seeded_engine`` fixture from conftest.py, which seeds a
fresh SQLite DB once per session deterministically.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.models import (
    Airport,
    BaggageRule,
    Booking,
    Customer,
    Flight,
    KBArticle,
    LoyaltyAccount,
    Refund,
    Seat,
    SupportMessage,
    SupportTicket,
)
from app.seed import SCALES


def test_seed_returns_expected_summary(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """seed() must return exact counts matching the 'small' preset."""
    _, summary = seeded_engine
    expected = SCALES["small"]
    for table, count in expected.items():
        assert summary[table] == count, f"{table}: got {summary[table]}, want {count}"


def test_row_counts_in_db(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """Every table in the small preset must have exactly the expected row count in the DB."""
    engine, _ = seeded_engine
    expected = SCALES["small"]
    with Session(engine) as s:
        assert s.execute(select(func.count()).select_from(Customer)).scalar_one() == expected["customers"]
        assert s.execute(select(func.count()).select_from(Airport)).scalar_one() == expected["airports"]
        assert s.execute(select(func.count()).select_from(Flight)).scalar_one() == expected["flights"]
        assert s.execute(select(func.count()).select_from(Booking)).scalar_one() == expected["bookings"]
        assert s.execute(select(func.count()).select_from(Seat)).scalar_one() == expected["seats"]
        assert s.execute(select(func.count()).select_from(BaggageRule)).scalar_one() == expected["baggage_rules"]
        assert s.execute(select(func.count()).select_from(Refund)).scalar_one() == expected["refunds"]
        assert s.execute(select(func.count()).select_from(SupportTicket)).scalar_one() == expected["support_tickets"]
        assert s.execute(select(func.count()).select_from(SupportMessage)).scalar_one() == expected["support_messages"]
        assert s.execute(select(func.count()).select_from(KBArticle)).scalar_one() == expected["kb_articles"]


def test_referential_integrity(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """All FKs in the seed must point to existing parents."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        # Every booking has a real customer + flight
        orphan_bookings = s.execute(
            select(func.count())
            .select_from(Booking)
            .outerjoin(Customer, Booking.customer_id == Customer.id)
            .where(Customer.id.is_(None))
        ).scalar_one()
        assert orphan_bookings == 0

        orphan_bookings_flight = s.execute(
            select(func.count())
            .select_from(Booking)
            .outerjoin(Flight, Booking.flight_id == Flight.id)
            .where(Flight.id.is_(None))
        ).scalar_one()
        assert orphan_bookings_flight == 0

        # Every refund has a real booking
        orphan_refunds = s.execute(
            select(func.count())
            .select_from(Refund)
            .outerjoin(Booking, Refund.booking_id == Booking.id)
            .where(Booking.id.is_(None))
        ).scalar_one()
        assert orphan_refunds == 0

        # Every support message has a real ticket
        orphan_messages = s.execute(
            select(func.count())
            .select_from(SupportMessage)
            .outerjoin(SupportTicket, SupportMessage.ticket_id == SupportTicket.id)
            .where(SupportTicket.id.is_(None))
        ).scalar_one()
        assert orphan_messages == 0

        # Every seat has a real flight
        orphan_seats = s.execute(
            select(func.count())
            .select_from(Seat)
            .outerjoin(Flight, Seat.flight_id == Flight.id)
            .where(Flight.id.is_(None))
        ).scalar_one()
        assert orphan_seats == 0


def test_sample_customer_has_full_graph(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """At least one customer should have bookings, support tickets, and loyalty."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        # Find customers that have BOTH a booking and a support ticket.
        rows = s.execute(
            select(Customer.id)
            .join(Booking, Booking.customer_id == Customer.id)
            .join(SupportTicket, SupportTicket.customer_id == Customer.id)
            .group_by(Customer.id)
            .limit(5)
        ).scalars().all()
        assert rows, "expected at least one customer with both a booking and a ticket"

        # Load one in full and check related collections.
        cust = s.get(Customer, rows[0])
        assert cust is not None
        assert len(cust.bookings) >= 1
        assert len(cust.support_tickets) >= 1
        # loyalty_account may or may not exist (70% rate), so just assert if present it's valid
        if cust.loyalty_account is not None:
            assert cust.loyalty_account.loyalty_number.startswith("LY-")


def test_ambiguous_ticket_booking_pairs_exist(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """The 50 deliberately ambiguous tickets must share their PNR with a same-customer booking."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        # Tickets numbered TKT-<pnr> where <pnr> is exactly 6 chars
        matches = s.execute(
            select(SupportTicket.ticket_number, SupportTicket.customer_id, Booking.booking_reference)
            .join(
                Booking,
                (Booking.customer_id == SupportTicket.customer_id)
                & (Booking.booking_reference == func.substr(SupportTicket.ticket_number, 5)),
            )
        ).all()
        assert len(matches) >= 50, f"expected >=50 ambiguous ticket/booking pairs, got {len(matches)}"


def test_status_mismatches_exist(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """Realistic ambiguity: some flights are cancelled/delayed but bookings remain confirmed."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        delayed_but_confirmed = s.execute(
            select(func.count())
            .select_from(Booking)
            .join(Flight, Booking.flight_id == Flight.id)
            .where(Flight.status == "delayed", Booking.booking_status == "confirmed")
        ).scalar_one()
        cancelled_but_confirmed = s.execute(
            select(func.count())
            .select_from(Booking)
            .join(Flight, Booking.flight_id == Flight.id)
            .where(Flight.status == "cancelled", Booking.booking_status == "confirmed")
        ).scalar_one()
        # At least one of each should exist in 1000 bookings — assert >= 1 to keep
        # the test deterministic regardless of RNG draw.
        assert delayed_but_confirmed >= 1
        assert cancelled_but_confirmed >= 1


def test_kb_articles_cover_required_categories(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """KB must contain articles for the topics the chatbot will need."""
    engine, _ = seeded_engine
    required = {"baggage", "refunds", "flight_change", "seats", "cancellation", "loyalty"}
    with Session(engine) as s:
        categories = set(s.execute(select(KBArticle.category).distinct()).scalars().all())
    missing = required - categories
    assert not missing, f"KB is missing categories: {missing}"


def test_loyalty_uniqueness(seeded_engine: tuple[Engine, dict[str, int]]) -> None:
    """Each loyalty account must belong to a unique customer."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        total = s.execute(select(func.count()).select_from(LoyaltyAccount)).scalar_one()
        distinct_customers = s.execute(
            select(func.count(func.distinct(LoyaltyAccount.customer_id)))
        ).scalar_one()
        assert total == distinct_customers
