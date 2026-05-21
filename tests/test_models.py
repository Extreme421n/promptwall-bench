"""Smoke tests verifying every model can be inserted and queried."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Airport,
    BaggageRule,
    Booking,
    ChatSession,
    Customer,
    Flight,
    KBArticle,
    LLMCall,
    LoyaltyAccount,
    Refund,
    Seat,
    SupportMessage,
    SupportTicket,
    ToolInvocation,
    Trace,
)


def test_models_import_cleanly() -> None:
    """Importing the model package should be side-effect free and complete."""
    from app import models  # noqa: F401

    # All declared classes are registered on the same metadata.
    assert "customers" in {t.name for t in models.Base.metadata.tables.values()}
    assert "tool_invocations" in {t.name for t in models.Base.metadata.tables.values()}
    # 15 core (Phase 1B) + 2 evaluation (Phase 2B)
    # + 1 promptwall_candidate_decisions (Phase 3A)
    # + 10 SaaS/billing (Phase B1)
    # + 11 commerce/orders (Phase B2)
    # + 7 textual knowledge (Phase 6B-1) = 46.
    assert len(models.Base.metadata.tables) == 46


def test_full_object_graph_insert_and_query(db: Session) -> None:
    """Insert a representative row in every table and assert relationships work."""
    # CRM
    customer = Customer(
        external_customer_id="CUST-0001",
        full_name="Ada Lovelace",
        email="ada@example.com",
        phone="+1-555-0100",
        segment="premium",
    )
    customer.loyalty_account = LoyaltyAccount(
        loyalty_number="LY-100001", tier="gold", points_balance=12500
    )
    db.add(customer)
    db.flush()

    # Airline
    jfk = Airport(code="JFK", city="New York", country="USA", timezone="America/New_York")
    lhr = Airport(code="LHR", city="London", country="UK", timezone="Europe/London")
    db.add_all([jfk, lhr])
    db.flush()

    dep = datetime(2026, 6, 1, 22, 0, tzinfo=timezone.utc)
    flight = Flight(
        flight_number="BA178",
        origin_airport_id=jfk.id,
        destination_airport_id=lhr.id,
        scheduled_departure=dep,
        scheduled_arrival=dep + timedelta(hours=7),
        status="scheduled",
        gate="B22",
    )
    db.add(flight)
    db.flush()

    db.add_all(
        [
            Seat(flight_id=flight.id, seat_number="12A", cabin_class="economy"),
            Seat(flight_id=flight.id, seat_number="2A", cabin_class="business", is_available=False),
        ]
    )

    booking = Booking(
        booking_reference="ABC123",
        customer_id=customer.id,
        flight_id=flight.id,
        booking_status="confirmed",
        cabin_class="economy",
        total_paid=Decimal("742.50"),
        currency="USD",
    )
    db.add(booking)
    db.flush()

    db.add(
        Refund(
            booking_id=booking.id,
            refund_status="pending",
            refund_amount=Decimal("100.00"),
            reason="schedule change",
            expected_resolution_date=date(2026, 6, 15),
        )
    )

    db.add(
        BaggageRule(
            route_type="international",
            cabin_class="economy",
            checked_bag_kg=23,
            cabin_bag_kg=7,
            policy_text="1 checked bag up to 23kg included on international economy.",
            effective_from=date(2026, 1, 1),
        )
    )

    # Support
    ticket = SupportTicket(
        ticket_number="TKT-0001",
        customer_id=customer.id,
        subject="Refund status question",
        status="open",
        priority="high",
    )
    ticket.messages = [
        SupportMessage(sender_type="customer", body="When will my refund arrive?"),
        SupportMessage(sender_type="agent", body="We're processing it."),
    ]
    db.add(ticket)

    # KB
    db.add(
        KBArticle(
            slug="checked-baggage-policy",
            title="Checked Baggage Policy",
            category="baggage",
            body="Economy passengers may check one bag up to 23kg on international routes.",
            version=1,
            is_active=True,
        )
    )

    # Observability
    chat = ChatSession(session_uuid="sess-001", customer_id=customer.id, channel="web")
    db.add(chat)
    db.flush()

    trace = Trace(
        session_id=chat.id,
        customer_id=customer.id,
        mode="baseline",
        user_message="What's the status of booking ABC123?",
        final_answer="Confirmed; departs Jun 1.",
        latency_ms=842,
        extra_metadata={"intent": "booking_status"},
    )
    db.add(trace)
    db.flush()

    db.add(
        LLMCall(
            trace_id=trace.id,
            provider="mock",
            model="mock-1",
            input_messages=[{"role": "user", "content": "..."}],
            output_message="ok",
            tool_calls_requested=[{"name": "get_booking", "arguments": {"ref": "ABC123"}}],
            prompt_tokens=42,
            completion_tokens=18,
            total_tokens=60,
            estimated_cost_usd=Decimal("0.000123"),
            latency_ms=540,
        )
    )
    db.add(
        ToolInvocation(
            trace_id=trace.id,
            tool_name="get_booking",
            input_json={"ref": "ABC123"},
            output_json={"status": "confirmed"},
            success=True,
            latency_ms=22,
            evidence_id="ev-001",
        )
    )

    db.flush()

    # --- Assertions ---
    fetched = db.execute(
        select(Customer).where(Customer.external_customer_id == "CUST-0001")
    ).scalar_one()
    assert fetched.loyalty_account is not None
    assert fetched.loyalty_account.tier == "gold"
    assert len(fetched.bookings) == 1
    assert fetched.bookings[0].booking_reference == "ABC123"
    assert len(fetched.support_tickets) == 1
    assert len(fetched.support_tickets[0].messages) == 2

    flight_back = db.execute(
        select(Flight).where(Flight.flight_number == "BA178")
    ).scalar_one()
    assert len(flight_back.seats) == 2
    assert flight_back.origin_airport.code == "JFK"
    assert flight_back.destination_airport.code == "LHR"

    trace_back = db.execute(select(Trace).where(Trace.mode == "baseline")).scalar_one()
    assert trace_back.extra_metadata == {"intent": "booking_status"}
    assert len(trace_back.llm_calls) == 1
    assert len(trace_back.tool_invocations) == 1
    assert trace_back.tool_invocations[0].evidence_id == "ev-001"


def test_unique_constraints(db: Session) -> None:
    """Booking reference must be unique."""
    import pytest
    from sqlalchemy.exc import IntegrityError

    c = Customer(external_customer_id="CUST-X", full_name="X", email="x@x.com")
    db.add(c)
    db.flush()

    a1 = Airport(code="AAA", city="A", country="A", timezone="UTC")
    a2 = Airport(code="BBB", city="B", country="B", timezone="UTC")
    db.add_all([a1, a2])
    db.flush()

    dep = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    f = Flight(
        flight_number="ZZ1",
        origin_airport_id=a1.id,
        destination_airport_id=a2.id,
        scheduled_departure=dep,
        scheduled_arrival=dep + timedelta(hours=2),
    )
    db.add(f)
    db.flush()

    db.add(
        Booking(
            booking_reference="DUP001",
            customer_id=c.id,
            flight_id=f.id,
            total_paid=Decimal("0"),
        )
    )
    db.flush()

    db.add(
        Booking(
            booking_reference="DUP001",  # same ref -> should fail
            customer_id=c.id,
            flight_id=f.id,
            total_paid=Decimal("0"),
        )
    )
    with pytest.raises(IntegrityError):
        db.flush()
