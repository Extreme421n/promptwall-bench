"""Deterministic seed data for the demo benchmark environment.

Supports three scale presets:

- ``small``  — the original Phase 1C dataset (~10k rows). Used by every test.
- ``medium`` — ~580k rows. Suitable for benchmark stress tests.
- ``large``  — ~3.8M rows + long support-message bodies, targeting ≥1GB on disk.

Implementation notes
--------------------

- Bulk inserts via ``session.execute(insert(Table), [dict, ...])``. Explicit
  primary keys are assigned up front so foreign keys are deterministic without
  a per-row round trip to the DB.
- Inserts are chunked (default 5000 rows per executemany call) so memory stays
  bounded for the ``large`` preset.
- Support-message bodies and KB article bodies scale with the preset so the
  ``large`` preset reaches ≥1GB without padding with meaningless blobs.
"""

from __future__ import annotations

import random
import string
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable, Optional

from faker import Faker
from sqlalchemy import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.models import (
    Airport,
    ApiUsageDaily,
    BaggageRule,
    Base,
    Booking,
    CommerceOrder,
    CommerceOrderItem,
    CommerceRefund,
    CommerceReturn,
    Customer,
    CustomerOrganization,
    Flight,
    InternalAgentNote,
    Invoice,
    InvoiceItem,
    KBArticle,
    LoyaltyAccount,
    OperationalIncident,
    Organization,
    OverageCharge,
    Plan,
    PolicyClause,
    PolicyDocument,
    Product,
    ProductAttribute,
    ProductCategory,
    ProductInventory,
    ProductPrice,
    ProductReturnRule,
    ProductWarrantyTerms,
    Refund,
    Seat,
    SeatAllocation,
    Shipment,
    Subscription,
    SupportMessage,
    SupportResolutionTemplate,
    SupportTicket,
    UsageEvent,
    Warehouse,
)

# ---------------------------------------------------------------------------
# Scale presets
# ---------------------------------------------------------------------------

SCALES: dict[str, dict[str, int]] = {
    "small": {
        # airline + support domain
        "customers": 500,
        "airports": 50,
        "flights": 300,
        "bookings": 1000,
        "seats": 3000,
        "baggage_rules": 200,
        "refunds": 300,
        "support_tickets": 800,
        "support_messages": 2500,
        "kb_articles": 50,
        # saas / billing domain (Phase B1)
        "organizations": 50,
        "customer_organizations": 150,
        "plans": 4,
        "subscriptions": 50,
        "invoices": 200,
        "invoice_items": 600,
        "usage_events": 500,
        "api_usage_daily": 1500,
        "seat_allocations": 50,
        "overage_charges": 80,
        # commerce / orders domain (Phase B2)
        "product_categories": 20,
        "products": 100,
        "product_attributes": 300,
        "product_prices": 200,
        "warehouses": 5,
        "product_inventory": 500,
        "commerce_orders": 300,
        "commerce_order_items": 900,
        "shipments": 280,
        "commerce_returns": 60,
        "commerce_refunds": 40,
        # textual knowledge (Phase 6B-2)
        "policy_documents": 50,
        "policy_clauses": 300,
        "product_warranty_terms": 100,
        "product_return_rules": 100,
        "internal_agent_notes": 500,
        "operational_incidents": 50,
        "support_resolution_templates": 100,
    },
    "medium": {
        "customers": 20_000,
        "airports": 50,
        "flights": 10_000,
        "bookings": 50_000,
        "seats": 200_000,
        "baggage_rules": 500,
        "refunds": 15_000,
        "support_tickets": 50_000,
        "support_messages": 250_000,
        "kb_articles": 1_000,
        # saas / billing
        "organizations": 2_000,
        "customer_organizations": 8_000,
        "plans": 4,
        "subscriptions": 2_000,
        "invoices": 8_000,
        "invoice_items": 24_000,
        "usage_events": 50_000,
        "api_usage_daily": 60_000,
        "seat_allocations": 2_000,
        "overage_charges": 3_000,
        # commerce / orders domain (Phase B2)
        "product_categories": 100,
        "products": 5_000,
        "product_attributes": 15_000,
        "product_prices": 10_000,
        "warehouses": 20,
        "product_inventory": 50_000,
        "commerce_orders": 30_000,
        "commerce_order_items": 90_000,
        "shipments": 28_000,
        "commerce_returns": 6_000,
        "commerce_refunds": 4_000,
        # textual knowledge (Phase 6B-2)
        "policy_documents": 500,
        "policy_clauses": 3_000,
        "product_warranty_terms": 5_000,
        "product_return_rules": 1_000,
        "internal_agent_notes": 50_000,
        "operational_incidents": 500,
        "support_resolution_templates": 500,
    },
    "large": {
        "customers": 200_000,
        "airports": 50,
        "flights": 50_000,
        "bookings": 500_000,
        "seats": 1_000_000,
        "baggage_rules": 1_000,
        "refunds": 50_000,
        "support_tickets": 200_000,
        "support_messages": 2_000_000,
        "kb_articles": 5_000,
        # saas / billing
        "organizations": 20_000,
        "customer_organizations": 80_000,
        "plans": 4,
        "subscriptions": 20_000,
        "invoices": 80_000,
        "invoice_items": 240_000,
        "usage_events": 500_000,
        "api_usage_daily": 600_000,
        "seat_allocations": 20_000,
        "overage_charges": 30_000,
        # commerce / orders domain (Phase B2)
        "product_categories": 500,
        "products": 50_000,
        "product_attributes": 150_000,
        "product_prices": 100_000,
        "warehouses": 50,
        "product_inventory": 500_000,
        "commerce_orders": 300_000,
        "commerce_order_items": 900_000,
        "shipments": 280_000,
        "commerce_returns": 60_000,
        "commerce_refunds": 40_000,
        # textual knowledge (Phase 6B-2)
        "policy_documents": 2_000,
        "policy_clauses": 15_000,
        "product_warranty_terms": 50_000,
        "product_return_rules": 5_000,
        "internal_agent_notes": 500_000,
        "operational_incidents": 5_000,
        "support_resolution_templates": 2_000,
    },
}

# How many support-message tokens we synthesize per row, by preset. Drives DB
# size without resorting to padding-style blobs.
_MESSAGE_SENTENCES: dict[str, tuple[int, int]] = {
    "small": (1, 1),
    "medium": (2, 5),
    "large": (5, 12),
}

_KB_SENTENCES: dict[str, tuple[int, int]] = {
    "small": (6, 8),
    "medium": (12, 18),
    "large": (30, 60),
}

# ---------------------------------------------------------------------------
# Reference data (unchanged from Phase 1C)
# ---------------------------------------------------------------------------

AIRPORTS: list[tuple[str, str, str, str]] = [
    ("JFK", "New York", "USA", "America/New_York"),
    ("LAX", "Los Angeles", "USA", "America/Los_Angeles"),
    ("ORD", "Chicago", "USA", "America/Chicago"),
    ("ATL", "Atlanta", "USA", "America/New_York"),
    ("DFW", "Dallas", "USA", "America/Chicago"),
    ("DEN", "Denver", "USA", "America/Denver"),
    ("SFO", "San Francisco", "USA", "America/Los_Angeles"),
    ("SEA", "Seattle", "USA", "America/Los_Angeles"),
    ("MIA", "Miami", "USA", "America/New_York"),
    ("BOS", "Boston", "USA", "America/New_York"),
    ("LHR", "London", "UK", "Europe/London"),
    ("LGW", "London", "UK", "Europe/London"),
    ("CDG", "Paris", "France", "Europe/Paris"),
    ("ORY", "Paris", "France", "Europe/Paris"),
    ("FRA", "Frankfurt", "Germany", "Europe/Berlin"),
    ("AMS", "Amsterdam", "Netherlands", "Europe/Amsterdam"),
    ("MAD", "Madrid", "Spain", "Europe/Madrid"),
    ("BCN", "Barcelona", "Spain", "Europe/Madrid"),
    ("FCO", "Rome", "Italy", "Europe/Rome"),
    ("ZRH", "Zurich", "Switzerland", "Europe/Zurich"),
    ("MUC", "Munich", "Germany", "Europe/Berlin"),
    ("VIE", "Vienna", "Austria", "Europe/Vienna"),
    ("BRU", "Brussels", "Belgium", "Europe/Brussels"),
    ("DUB", "Dublin", "Ireland", "Europe/Dublin"),
    ("CPH", "Copenhagen", "Denmark", "Europe/Copenhagen"),
    ("ARN", "Stockholm", "Sweden", "Europe/Stockholm"),
    ("OSL", "Oslo", "Norway", "Europe/Oslo"),
    ("HEL", "Helsinki", "Finland", "Europe/Helsinki"),
    ("IST", "Istanbul", "Turkey", "Europe/Istanbul"),
    ("DXB", "Dubai", "UAE", "Asia/Dubai"),
    ("DOH", "Doha", "Qatar", "Asia/Qatar"),
    ("SIN", "Singapore", "Singapore", "Asia/Singapore"),
    ("HKG", "Hong Kong", "Hong Kong", "Asia/Hong_Kong"),
    ("NRT", "Tokyo", "Japan", "Asia/Tokyo"),
    ("HND", "Tokyo", "Japan", "Asia/Tokyo"),
    ("ICN", "Seoul", "South Korea", "Asia/Seoul"),
    ("PEK", "Beijing", "China", "Asia/Shanghai"),
    ("PVG", "Shanghai", "China", "Asia/Shanghai"),
    ("BKK", "Bangkok", "Thailand", "Asia/Bangkok"),
    ("KUL", "Kuala Lumpur", "Malaysia", "Asia/Kuala_Lumpur"),
    ("DEL", "Delhi", "India", "Asia/Kolkata"),
    ("BOM", "Mumbai", "India", "Asia/Kolkata"),
    ("SYD", "Sydney", "Australia", "Australia/Sydney"),
    ("MEL", "Melbourne", "Australia", "Australia/Melbourne"),
    ("AKL", "Auckland", "New Zealand", "Pacific/Auckland"),
    ("YYZ", "Toronto", "Canada", "America/Toronto"),
    ("YVR", "Vancouver", "Canada", "America/Vancouver"),
    ("GRU", "Sao Paulo", "Brazil", "America/Sao_Paulo"),
    ("MEX", "Mexico City", "Mexico", "America/Mexico_City"),
    ("JNB", "Johannesburg", "South Africa", "Africa/Johannesburg"),
]

AIRLINE_CODES = ["BA", "AA", "DL", "UA", "LH", "AF", "KL", "QR", "EK", "SQ", "NH", "QF"]
CABIN_CLASSES = ["economy", "premium_economy", "business", "first"]
CABIN_WEIGHTS = [0.70, 0.15, 0.12, 0.03]
TICKET_STATUSES = ["open", "pending", "resolved", "closed"]
TICKET_STATUS_WEIGHTS = [0.30, 0.25, 0.25, 0.20]
TICKET_PRIORITIES = ["low", "normal", "high", "urgent"]
TICKET_PRIORITY_WEIGHTS = [0.15, 0.55, 0.22, 0.08]
REFUND_STATUSES = ["pending", "approved", "rejected", "completed"]
REFUND_STATUS_WEIGHTS = [0.35, 0.20, 0.10, 0.35]
CUSTOMER_SEGMENTS = ["standard", "frequent", "premium", "corporate"]
CUSTOMER_SEGMENT_WEIGHTS = [0.60, 0.20, 0.15, 0.05]

TICKET_SUBJECTS = [
    "Refund status for booking",
    "Baggage allowance question",
    "Flight delay compensation",
    "Seat selection issue",
    "Loyalty points missing",
    "Cancellation policy clarification",
    "Change flight date",
    "Special meal request",
    "Wheelchair assistance",
    "Lost baggage claim",
    "Upgrade request",
    "Itinerary change",
    "Refund taking too long",
    "Misspelled name on booking",
    "Cannot check in online",
]

REFUND_REASONS = [
    "schedule change",
    "customer request",
    "cancellation policy",
    "double booking",
    "weather disruption",
    "operational cancellation",
]

CUSTOMER_MESSAGE_TEMPLATES = [
    "Hi, I have a question about {subject}.",
    "Following up — any update on this?",
    "Can someone please look into this? It's been a few days.",
    "Thanks for the quick response.",
    "That doesn't fully answer my question.",
    "Could you also check my loyalty points?",
]
AGENT_MESSAGE_TEMPLATES = [
    "Hi {name}, thanks for reaching out. I'm looking into this now.",
    "I've escalated this to our refunds team — expect an update within 5-7 business days.",
    "I can confirm that your request has been processed.",
    "Could you please share your booking reference so I can look this up?",
    "Apologies for the delay. We are investigating with our operations team.",
]

KB_TOPICS: list[tuple[str, str, str]] = [
    ("baggage", "Checked Baggage Policy", "Economy passengers may check one bag up to 23kg on international routes; domestic flights allow one bag up to 18kg."),
    ("baggage", "Cabin Baggage Allowance", "Cabin bags must fit in the overhead bin: 55x40x20cm and under 7kg for economy."),
    ("baggage", "Excess Baggage Fees", "Excess baggage fees apply per kilogram above the included allowance and vary by route."),
    ("baggage", "Lost or Damaged Baggage Claim", "Lost or damaged baggage must be reported within 7 days at the airport baggage desk."),
    ("refunds", "Refund Eligibility", "Refunds are eligible when the airline cancels a flight or under refundable fare conditions."),
    ("refunds", "How to Request a Refund", "Refunds can be requested via the support portal using your booking reference."),
    ("refunds", "Refund Processing Times", "Refunds typically take 5-7 business days; payments to original payment methods may take 1-2 billing cycles to appear."),
    ("refunds", "Non-refundable Fares", "Saver, Basic, and Promo fares are non-refundable but may be eligible for travel credit."),
    ("flight_change", "Changing Your Flight Date", "Flight date changes can be made up to 24 hours before departure; fees apply per fare class."),
    ("flight_change", "Same-day Flight Changes", "Same-day standby is available to elite tier members at no charge."),
    ("flight_change", "Cancellation and Rebooking", "If your flight is cancelled by the airline you will be rebooked on the next available flight at no charge."),
    ("seats", "Selecting a Seat", "Seat selection is included for premium fares and elite members; otherwise a fee applies."),
    ("seats", "Extra Legroom Seats", "Extra legroom seats can be purchased at booking or check-in subject to availability."),
    ("seats", "Seat Changes at the Airport", "Seat changes at the airport are subject to availability and check-in window cutoffs."),
    ("cancellation", "Cancellation Policy", "Cancellations are subject to the fare rules of the booking class purchased."),
    ("cancellation", "Involuntary Cancellation", "If we cancel your flight you may choose a full refund or a free rebooking."),
    ("loyalty", "Earning Points", "You earn loyalty points based on the fare paid and distance flown."),
    ("loyalty", "Redeeming Points", "Points can be redeemed for flights, upgrades, and partner products via the loyalty portal."),
    ("loyalty", "Loyalty Tiers", "The program has standard, silver, gold, and platinum tiers with cumulative annual benefits."),
    ("loyalty", "Missing Points Inquiry", "Missing points can be claimed up to 6 months after the flight via the missing-credit form."),
    ("check_in", "Online Check-in", "Online check-in opens 24 hours before departure and closes 60 minutes before."),
    ("check_in", "Airport Check-in Cutoffs", "Airport check-in closes 45 minutes before domestic and 60 minutes before international departures."),
    ("special_assistance", "Wheelchair Assistance", "Wheelchair assistance must be requested at least 48 hours before departure."),
    ("special_assistance", "Traveling with Infants", "Infants under 2 travel on an adult's lap free domestically; international fees may apply."),
    ("special_assistance", "Special Meals", "Special meals must be requested at least 24 hours before departure."),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_pnr(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))


def _bulk_insert(
    session: Session,
    table: Any,
    rows: list[dict[str, Any]],
    *,
    chunk_size: int = 5000,
) -> None:
    """Bulk-insert via executemany, chunked to bound memory + statement size."""
    if not rows:
        return
    stmt = insert(table)
    for i in range(0, len(rows), chunk_size):
        session.execute(stmt, rows[i : i + chunk_size])


def reset_schema(engine: Engine) -> None:
    """Drop and recreate every table on ``Base.metadata``."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def seed(
    engine: Engine,
    scale: str = "small",
    *,
    seed_value: int = 42,
    verbose: bool | None = None,
) -> dict[str, int]:
    """Populate the DB with a self-consistent realistic dataset.

    Returns a dict of row counts per table.

    Progress is logged on stderr for the ``medium`` and ``large`` presets
    (``verbose`` defaults to True for those, False for ``small``). Pass
    ``verbose=True/False`` to override.
    """
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; expected one of {list(SCALES)}")

    counts = SCALES[scale]
    rng = random.Random(seed_value)
    fake = Faker()
    Faker.seed(seed_value)
    if verbose is None:
        verbose = scale in ("medium", "large")
    _log_start = time.perf_counter()

    def _section(label: str, n: int, started: float) -> None:
        if verbose:
            dt = time.perf_counter() - started
            elapsed = time.perf_counter() - _log_start
            print(
                f"  [seed] {label:<22} {n:>10,} rows  in {dt:6.2f}s "
                f"(+{elapsed:6.2f}s total)",
                flush=True,
            )

    summary: dict[str, int] = {}

    with Session(engine, expire_on_commit=False) as session:
        # ---- Airports ----
        airports = [
            {"id": i + 1, "code": code, "city": city, "country": country, "timezone": tz}
            for i, (code, city, country, tz) in enumerate(AIRPORTS[: counts["airports"]])
        ]
        _t0 = time.perf_counter()
        _bulk_insert(session, Airport, airports)
        airport_ids = [a["id"] for a in airports]
        summary["airports"] = len(airports)
        _section("airports", len(airports), _t0)

        # ---- Customers + loyalty ----
        _t0 = time.perf_counter()
        customer_rows: list[dict[str, Any]] = []
        loyalty_rows: list[dict[str, Any]] = []
        loyalty_seq = 100001
        loyalty_id = 1
        for i in range(counts["customers"]):
            cid = i + 1
            segment = rng.choices(CUSTOMER_SEGMENTS, weights=CUSTOMER_SEGMENT_WEIGHTS)[0]
            customer_rows.append(
                {
                    "id": cid,
                    "external_customer_id": f"CUST-{cid:05d}",
                    "full_name": fake.name(),
                    "email": fake.unique.email(),
                    "phone": fake.phone_number()[:40],
                    "segment": segment,
                }
            )
            if rng.random() < 0.70:
                if segment in ("premium", "corporate"):
                    tier = rng.choice(["gold", "platinum"])
                elif segment == "frequent":
                    tier = rng.choice(["silver", "gold"])
                else:
                    tier = rng.choice(["standard", "silver"])
                loyalty_rows.append(
                    {
                        "id": loyalty_id,
                        "customer_id": cid,
                        "loyalty_number": f"LY-{loyalty_seq}",
                        "tier": tier,
                        "points_balance": rng.randint(0, 250_000),
                    }
                )
                loyalty_seq += 1
                loyalty_id += 1
        _bulk_insert(session, Customer, customer_rows)
        _bulk_insert(session, LoyaltyAccount, loyalty_rows)
        customer_ids = [c["id"] for c in customer_rows]
        cust_first_names: dict[int, str] = {
            c["id"]: c["full_name"].split()[0] for c in customer_rows
        }
        summary["customers"] = len(customer_rows)
        summary["loyalty_accounts"] = len(loyalty_rows)
        _section("customers + loyalty", len(customer_rows) + len(loyalty_rows), _t0)

        # ---- Flights ----
        _t0 = time.perf_counter()
        flight_rows: list[dict[str, Any]] = []
        flight_statuses: dict[int, str] = {}
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        for i in range(counts["flights"]):
            fid = i + 1
            o = rng.choice(airport_ids)
            d = rng.choice(airport_ids)
            while d == o:
                d = rng.choice(airport_ids)
            days_offset = rng.randint(-30, 60)
            hour = rng.randint(0, 23)
            minute = rng.choice([0, 15, 30, 45])
            dep = (now + timedelta(days=days_offset)).replace(hour=hour, minute=minute)
            arr = dep + timedelta(
                hours=rng.randint(1, 14), minutes=rng.choice([0, 15, 30, 45])
            )
            if days_offset < -1:
                status = rng.choices(["arrived", "cancelled"], weights=[0.95, 0.05])[0]
            elif days_offset == 0:
                status = rng.choices(
                    ["boarding", "departed", "delayed", "scheduled"],
                    weights=[0.20, 0.30, 0.20, 0.30],
                )[0]
            else:
                status = rng.choices(
                    ["scheduled", "delayed", "cancelled"], weights=[0.88, 0.07, 0.05]
                )[0]
            flight_statuses[fid] = status
            flight_rows.append(
                {
                    "id": fid,
                    "flight_number": f"{rng.choice(AIRLINE_CODES)}{rng.randint(100, 9999)}",
                    "origin_airport_id": o,
                    "destination_airport_id": d,
                    "scheduled_departure": dep,
                    "scheduled_arrival": arr,
                    "status": status,
                    "gate": (
                        f"{rng.choice('ABCDEFGH')}{rng.randint(1, 40)}"
                        if status != "cancelled"
                        else None
                    ),
                }
            )
        _bulk_insert(session, Flight, flight_rows)
        flight_ids = [f["id"] for f in flight_rows]
        summary["flights"] = len(flight_rows)
        _section("flights", len(flight_rows), _t0)

        # ---- Seats ----
        _t0 = time.perf_counter()
        seat_rows: list[dict[str, Any]] = []
        target_seats = counts["seats"]
        per_flight = target_seats // len(flight_ids)
        extras = target_seats - per_flight * len(flight_ids)
        seat_id = 1
        for f_idx, fid in enumerate(flight_ids):
            n = per_flight + (1 if f_idx < extras else 0)
            for s_idx in range(n):
                row = (s_idx // 6) + 1
                letter = "ABCDEF"[s_idx % 6]
                if row <= 1:
                    cabin = "first"
                elif row <= 3:
                    cabin = "business"
                elif row <= 5:
                    cabin = "premium_economy"
                else:
                    cabin = "economy"
                seat_rows.append(
                    {
                        "id": seat_id,
                        "flight_id": fid,
                        "seat_number": f"{row}{letter}",
                        "cabin_class": cabin,
                        "is_available": rng.random() < 0.40,
                    }
                )
                seat_id += 1
        _bulk_insert(session, Seat, seat_rows)
        summary["seats"] = len(seat_rows)
        _section("seats", len(seat_rows), _t0)

        # ---- Bookings (with 50 deliberately-ambiguous PNRs) ----
        _t0 = time.perf_counter()
        ambig_pnrs: list[str] = []
        used_refs: set[str] = set()
        while len(ambig_pnrs) < 50:
            p = _rand_pnr(rng)
            if p not in used_refs:
                used_refs.add(p)
                ambig_pnrs.append(p)
        ambig_customer_ids = rng.sample(customer_ids, 50)
        ambig_map = dict(zip(ambig_customer_ids, ambig_pnrs))

        cabin_price_base = {"economy": 200, "premium_economy": 500, "business": 1500, "first": 4500}
        booking_rows: list[dict[str, Any]] = []
        for i in range(counts["bookings"]):
            bid = i + 1
            if i < len(ambig_pnrs):
                cust = ambig_customer_ids[i]
                ref = ambig_pnrs[i]
            else:
                cust = customer_ids[rng.randrange(len(customer_ids))]
                while True:
                    candidate = _rand_pnr(rng)
                    if candidate not in used_refs:
                        used_refs.add(candidate)
                        ref = candidate
                        break
            fid = flight_ids[rng.randrange(len(flight_ids))]
            cabin = rng.choices(CABIN_CLASSES, weights=CABIN_WEIGHTS)[0]
            f_status = flight_statuses[fid]
            if f_status == "cancelled":
                bstatus = rng.choices(
                    ["cancelled", "refunded", "confirmed"], weights=[0.55, 0.30, 0.15]
                )[0]
            elif f_status == "delayed":
                bstatus = rng.choices(
                    ["confirmed", "pending", "cancelled"], weights=[0.85, 0.10, 0.05]
                )[0]
            else:
                bstatus = rng.choices(
                    ["confirmed", "pending", "cancelled", "refunded"],
                    weights=[0.78, 0.10, 0.07, 0.05],
                )[0]
            amount = Decimal(str(cabin_price_base[cabin] + rng.randint(-50, 400))).quantize(
                Decimal("0.01")
            )
            booking_rows.append(
                {
                    "id": bid,
                    "booking_reference": ref,
                    "customer_id": cust,
                    "flight_id": fid,
                    "booking_status": bstatus,
                    "cabin_class": cabin,
                    "total_paid": amount,
                    "currency": rng.choices(["USD", "EUR", "GBP"], weights=[0.70, 0.20, 0.10])[0],
                }
            )
        _bulk_insert(session, Booking, booking_rows)
        summary["bookings"] = len(booking_rows)
        _section("bookings", len(booking_rows), _t0)

        # ---- Baggage rules ----
        route_types = ["domestic", "intra-continental", "international", "ultra-long-haul"]
        baggage_rows: list[dict[str, Any]] = []
        target_rules = counts["baggage_rules"]
        per_combo = target_rules // (len(route_types) * len(CABIN_CLASSES))
        remainder = target_rules - per_combo * len(route_types) * len(CABIN_CLASSES)
        rule_id = 1
        for r in route_types:
            for cabin in CABIN_CLASSES:
                versions = per_combo + (1 if remainder > 0 else 0)
                if remainder > 0:
                    remainder -= 1
                base_checked = {"economy": 23, "premium_economy": 23, "business": 32, "first": 32}[cabin]
                cabin_bag = {"economy": 7, "premium_economy": 10, "business": 12, "first": 15}[cabin]
                if r == "domestic":
                    base_checked = max(0, base_checked - 5)
                for v in range(versions):
                    eff = date(2018 + (v % 8), rng.randint(1, 12), rng.randint(1, 28))
                    baggage_rows.append(
                        {
                            "id": rule_id,
                            "route_type": r,
                            "cabin_class": cabin,
                            "checked_bag_kg": base_checked,
                            "cabin_bag_kg": cabin_bag,
                            "policy_text": (
                                f"{cabin.replace('_', ' ').title()} passengers on {r} flights "
                                f"may check one bag up to {base_checked}kg and one cabin bag up "
                                f"to {cabin_bag}kg. Effective {eff.isoformat()}."
                            ),
                            "effective_from": eff,
                        }
                    )
                    rule_id += 1
        _bulk_insert(session, BaggageRule, baggage_rows)
        summary["baggage_rules"] = len(baggage_rows)

        # ---- Refunds ----
        _t0 = time.perf_counter()
        refund_target_ids = rng.sample(
            [b["id"] for b in booking_rows], min(counts["refunds"], len(booking_rows))
        )
        booking_amount_by_id = {b["id"]: b["total_paid"] for b in booking_rows}
        refund_rows: list[dict[str, Any]] = []
        for i, b_id in enumerate(refund_target_ids):
            status = rng.choices(REFUND_STATUSES, weights=REFUND_STATUS_WEIGHTS)[0]
            ratio = Decimal(str(round(rng.uniform(0.3, 1.0), 2)))
            amount = (booking_amount_by_id[b_id] * ratio).quantize(Decimal("0.01"))
            refund_rows.append(
                {
                    "id": i + 1,
                    "booking_id": b_id,
                    "refund_status": status,
                    "refund_amount": amount,
                    "reason": rng.choice(REFUND_REASONS),
                    "expected_resolution_date": date.today()
                    + timedelta(days=rng.randint(-30, 30)),
                }
            )
        _bulk_insert(session, Refund, refund_rows)
        summary["refunds"] = len(refund_rows)
        _section("refunds", len(refund_rows), _t0)

        # ---- Support tickets (with deliberate ambiguity) ----
        _t0 = time.perf_counter()
        ticket_rows: list[dict[str, Any]] = []
        used_ticket_numbers: set[str] = set()
        ticket_id = 1
        for cust_id, pnr in ambig_map.items():
            number = f"TKT-{pnr}"
            used_ticket_numbers.add(number)
            ticket_rows.append(
                {
                    "id": ticket_id,
                    "ticket_number": number,
                    "customer_id": cust_id,
                    "subject": f"{rng.choice(TICKET_SUBJECTS)} {pnr}",
                    "status": rng.choices(TICKET_STATUSES, weights=TICKET_STATUS_WEIGHTS)[0],
                    "priority": rng.choices(TICKET_PRIORITIES, weights=TICKET_PRIORITY_WEIGHTS)[0],
                }
            )
            ticket_id += 1
        while len(ticket_rows) < counts["support_tickets"]:
            cust_id = customer_ids[rng.randrange(len(customer_ids))]
            while True:
                candidate = f"TKT-{_rand_pnr(rng)}"
                if candidate not in used_ticket_numbers:
                    used_ticket_numbers.add(candidate)
                    break
            ticket_rows.append(
                {
                    "id": ticket_id,
                    "ticket_number": candidate,
                    "customer_id": cust_id,
                    "subject": rng.choice(TICKET_SUBJECTS),
                    "status": rng.choices(TICKET_STATUSES, weights=TICKET_STATUS_WEIGHTS)[0],
                    "priority": rng.choices(TICKET_PRIORITIES, weights=TICKET_PRIORITY_WEIGHTS)[0],
                }
            )
            ticket_id += 1
        _bulk_insert(session, SupportTicket, ticket_rows)
        summary["support_tickets"] = len(ticket_rows)
        _section("support_tickets", len(ticket_rows), _t0)

        # ---- Support messages ----
        _t0 = time.perf_counter()
        total_messages = counts["support_messages"]
        n_tickets = len(ticket_rows)
        per_ticket = [max(1, int(rng.gauss(3.1, 1.2))) for _ in range(n_tickets)]
        diff = total_messages - sum(per_ticket)
        if diff > 0:
            for i in range(diff):
                per_ticket[i % n_tickets] += 1
        elif diff < 0:
            i = 0
            while diff < 0:
                if per_ticket[i % n_tickets] > 1:
                    per_ticket[i % n_tickets] -= 1
                    diff += 1
                i += 1

        msg_min, msg_max = _MESSAGE_SENTENCES[scale]
        message_rows: list[dict[str, Any]] = []
        msg_id = 1
        for t, n in zip(ticket_rows, per_ticket):
            for j in range(n):
                if j == 0:
                    sender = "customer"
                else:
                    sender = rng.choices(
                        ["agent", "customer", "bot"], weights=[0.50, 0.40, 0.10]
                    )[0]
                first_name = cust_first_names.get(t["customer_id"], "there")
                body = _build_message_body(
                    fake=fake,
                    rng=rng,
                    scale=scale,
                    sender=sender,
                    subject=t["subject"],
                    first_name=first_name,
                    min_s=msg_min,
                    max_s=msg_max,
                )
                message_rows.append(
                    {
                        "id": msg_id,
                        "ticket_id": t["id"],
                        "sender_type": sender,
                        "body": body,
                    }
                )
                msg_id += 1

                if len(message_rows) >= 10_000:
                    _bulk_insert(session, SupportMessage, message_rows)
                    message_rows = []
        if message_rows:
            _bulk_insert(session, SupportMessage, message_rows)
        summary["support_messages"] = total_messages
        _section("support_messages", total_messages, _t0)

        # ---- KB articles ----
        _t0 = time.perf_counter()
        kb_min, kb_max = _KB_SENTENCES[scale]
        kb_rows: list[dict[str, Any]] = []
        used_slugs: set[str] = set()
        for k, (cat, title, body) in enumerate(KB_TOPICS):
            slug = f"{cat}-{k + 1}"
            used_slugs.add(slug)
            kb_rows.append(
                {
                    "id": k + 1,
                    "slug": slug,
                    "title": title,
                    "category": cat,
                    "body": (
                        f"{body} "
                        + fake.paragraph(nb_sentences=rng.randint(kb_min, kb_max))
                    ),
                    "version": rng.randint(1, 5),
                    "is_active": True,
                }
            )
        categories = sorted({c for c, _, _ in KB_TOPICS})
        extra_idx = 0
        next_id = len(kb_rows) + 1
        while len(kb_rows) < counts["kb_articles"]:
            cat = rng.choice(categories)
            extra_idx += 1
            slug = f"{cat}-faq-{extra_idx}"
            if slug in used_slugs:
                continue
            used_slugs.add(slug)
            kb_rows.append(
                {
                    "id": next_id,
                    "slug": slug,
                    "title": f"{cat.replace('_', ' ').title()} FAQ #{extra_idx}",
                    "category": cat,
                    "body": fake.paragraph(nb_sentences=rng.randint(kb_min, kb_max)),
                    "version": rng.randint(1, 5),
                    "is_active": rng.random() < 0.95,
                }
            )
            next_id += 1
        _bulk_insert(session, KBArticle, kb_rows)
        summary["kb_articles"] = len(kb_rows)
        _section("kb_articles", len(kb_rows), _t0)

        # -------------------------------------------------------------
        # SaaS / billing domain (Phase B1) — added at the end so all
        # prior RNG draws (and therefore airline/support data) are
        # unchanged. Any new randomness only affects SaaS rows.
        # -------------------------------------------------------------
        _t0 = time.perf_counter()
        _seed_saas(
            session=session,
            counts=counts,
            customer_ids=customer_ids,
            rng=rng,
            fake=fake,
            summary=summary,
        )
        _section(
            "saas/billing",
            sum(
                summary.get(k, 0)
                for k in (
                    "organizations",
                    "customer_organizations",
                    "plans",
                    "subscriptions",
                    "invoices",
                    "invoice_items",
                    "usage_events",
                    "api_usage_daily",
                    "seat_allocations",
                    "overage_charges",
                )
            ),
            _t0,
        )

        # -------------------------------------------------------------
        # Commerce / orders domain (Phase B2) — added last so all RNG
        # draws for airline + SaaS are unchanged.
        # -------------------------------------------------------------
        _t0 = time.perf_counter()
        _seed_commerce(
            session=session,
            counts=counts,
            customer_ids=customer_ids,
            rng=rng,
            fake=fake,
            summary=summary,
        )
        _section(
            "commerce/orders",
            sum(
                summary.get(k, 0)
                for k in (
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
            ),
            _t0,
        )

        # -------------------------------------------------------------
        # Textual knowledge (Phase 6B-2) — runs last so all referenced
        # IDs (products, categories, customers) already exist.
        # -------------------------------------------------------------
        _t0 = time.perf_counter()
        _seed_knowledge(
            session=session,
            counts=counts,
            customer_ids=customer_ids,
            rng=rng,
            fake=fake,
            summary=summary,
        )
        _section(
            "knowledge/text",
            sum(
                summary.get(k, 0)
                for k in (
                    "policy_documents",
                    "policy_clauses",
                    "product_warranty_terms",
                    "product_return_rules",
                    "internal_agent_notes",
                    "operational_incidents",
                    "support_resolution_templates",
                )
            ),
            _t0,
        )

        if verbose:
            print(f"  [seed] committing transaction...", flush=True)
        _commit_t0 = time.perf_counter()
        session.commit()
        if verbose:
            dt = time.perf_counter() - _commit_t0
            total_dt = time.perf_counter() - _log_start
            print(
                f"  [seed] commit completed in {dt:.2f}s (total seed time: {total_dt:.2f}s)",
                flush=True,
            )

    return summary


# ---------------------------------------------------------------------------
# Support-message body synthesis (scale-aware)
# ---------------------------------------------------------------------------


def _build_message_body(
    *,
    fake: Faker,
    rng: random.Random,
    scale: str,
    sender: str,
    subject: str,
    first_name: str,
    min_s: int,
    max_s: int,
) -> str:
    """Generate a sender-appropriate message body sized for ``scale``."""
    if scale == "small":
        if sender == "agent":
            return rng.choice(AGENT_MESSAGE_TEMPLATES).format(name=first_name)
        if sender == "bot":
            return "Auto-reply: We have received your message and will respond shortly."
        return rng.choice(CUSTOMER_MESSAGE_TEMPLATES).format(subject=subject.lower())

    # Medium and large: prepend a sender-shaped opener, then append several
    # sentences of realistic prose so the dataset reaches the target size
    # without padding-style blobs.
    if sender == "agent":
        opener = rng.choice(AGENT_MESSAGE_TEMPLATES).format(name=first_name)
    elif sender == "bot":
        opener = (
            "Auto-reply: thank you for contacting us. Your message has been "
            "logged and will be reviewed by an agent."
        )
    else:
        opener = rng.choice(CUSTOMER_MESSAGE_TEMPLATES).format(subject=subject.lower())

    body_text = fake.paragraph(nb_sentences=rng.randint(min_s, max_s))
    return f"{opener} {body_text}"


# ---------------------------------------------------------------------------
# SaaS / billing seeding (Phase B1)
# ---------------------------------------------------------------------------

# Static reference data: 4 plan tiers used across every preset.
_PLAN_CATALOG: list[dict[str, Any]] = [
    {
        "name": "Starter",
        "tier": "starter",
        "monthly_price": Decimal("29.00"),
        "included_seats": 5,
        "included_api_calls": 10_000,
        "overage_price_per_1000_calls": Decimal("0.5000"),
        "is_active": True,
    },
    {
        "name": "Pro",
        "tier": "pro",
        "monthly_price": Decimal("99.00"),
        "included_seats": 25,
        "included_api_calls": 100_000,
        "overage_price_per_1000_calls": Decimal("0.3000"),
        "is_active": True,
    },
    {
        "name": "Business",
        "tier": "business",
        "monthly_price": Decimal("399.00"),
        "included_seats": 100,
        "included_api_calls": 1_000_000,
        "overage_price_per_1000_calls": Decimal("0.2000"),
        "is_active": True,
    },
    {
        "name": "Enterprise",
        "tier": "enterprise",
        "monthly_price": Decimal("1500.00"),
        "included_seats": 500,
        "included_api_calls": 10_000_000,
        "overage_price_per_1000_calls": Decimal("0.1000"),
        "is_active": True,
    },
]

_SUBSCRIPTION_STATUSES = ["active", "past_due", "trialing", "canceled"]
_SUBSCRIPTION_STATUS_WEIGHTS = [0.80, 0.05, 0.05, 0.10]
_INVOICE_STATUSES = ["paid", "issued", "overdue", "draft", "void"]
_INVOICE_STATUS_WEIGHTS = [0.65, 0.18, 0.10, 0.04, 0.03]
_USAGE_EVENT_TYPES = ["api_call", "seat_added", "seat_removed", "feature_use", "login"]
_USAGE_EVENT_WEIGHTS = [0.60, 0.05, 0.03, 0.20, 0.12]
_MEMBER_ROLES = ["owner", "admin", "member", "billing", "viewer"]
_MEMBER_ROLE_WEIGHTS = [0.10, 0.15, 0.55, 0.05, 0.15]
_OVERAGE_REASONS = [
    "API call volume exceeded plan allowance",
    "Additional seats added mid-cycle",
    "Premium feature usage above included tier",
    "Burst usage during incident response",
]
_INVOICE_LINE_TEMPLATES = [
    "Monthly subscription — {plan} plan",
    "Additional seats ({n})",
    "API overage ({n} calls × {rate})",
    "Pro-rated upgrade adjustment",
    "Support package — premium hours",
    "Annual discount credit",
]


def _seed_saas(
    *,
    session: Session,
    counts: dict[str, int],
    customer_ids: list[int],
    rng: random.Random,
    fake: Faker,
    summary: dict[str, int],
) -> None:
    """Seed the SaaS/billing tables.

    Determinism: this function only consumes new ``rng`` draws and a fresh
    ``Faker.unique.company()`` stream. The airline/support seed code above
    runs before this function and is unchanged by it.
    """
    # ---- plans (fixed catalog) ----
    plan_count = min(counts["plans"], len(_PLAN_CATALOG))
    plan_rows = [
        {"id": i + 1, **p} for i, p in enumerate(_PLAN_CATALOG[:plan_count])
    ]
    _bulk_insert(session, Plan, plan_rows)
    plan_ids = [p["id"] for p in plan_rows]
    summary["plans"] = len(plan_rows)

    # ---- organizations ----
    org_rows: list[dict[str, Any]] = []
    for i in range(counts["organizations"]):
        oid = i + 1
        org_rows.append(
            {
                "id": oid,
                "name": fake.unique.company(),
                "external_org_id": f"ORG-{oid:05d}",
            }
        )
    _bulk_insert(session, Organization, org_rows)
    org_ids = [o["id"] for o in org_rows]
    summary["organizations"] = len(org_rows)

    # ---- customer_organizations (membership) ----
    # We don't oversample: pick distinct (customer_id, organization_id) pairs.
    target_links = min(
        counts["customer_organizations"],
        len(customer_ids) * len(org_ids),
    )
    seen_pairs: set[tuple[int, int]] = set()
    co_rows: list[dict[str, Any]] = []
    while len(co_rows) < target_links:
        c = customer_ids[rng.randrange(len(customer_ids))]
        o = org_ids[rng.randrange(len(org_ids))]
        if (c, o) in seen_pairs:
            continue
        seen_pairs.add((c, o))
        co_rows.append(
            {
                "id": len(co_rows) + 1,
                "customer_id": c,
                "organization_id": o,
                "role": rng.choices(_MEMBER_ROLES, weights=_MEMBER_ROLE_WEIGHTS)[0],
            }
        )
    _bulk_insert(session, CustomerOrganization, co_rows)
    summary["customer_organizations"] = len(co_rows)

    # ---- subscriptions (1 per org until count is reached) ----
    sub_target = min(counts["subscriptions"], len(org_ids))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    sub_rows: list[dict[str, Any]] = []
    org_to_sub: dict[int, dict[str, Any]] = {}
    for i, oid in enumerate(org_ids[:sub_target]):
        sid = i + 1
        status = rng.choices(
            _SUBSCRIPTION_STATUSES, weights=_SUBSCRIPTION_STATUS_WEIGHTS
        )[0]
        started = now - timedelta(days=rng.randint(30, 720))
        renews = now + timedelta(days=rng.randint(1, 30))
        canceled = (
            started + timedelta(days=rng.randint(30, 600)) if status == "canceled" else None
        )
        plan_id = rng.choice(plan_ids)
        row = {
            "id": sid,
            "organization_id": oid,
            "plan_id": plan_id,
            "status": status,
            "started_at": started,
            "renews_at": renews,
            "canceled_at": canceled,
        }
        sub_rows.append(row)
        org_to_sub[oid] = row
    _bulk_insert(session, Subscription, sub_rows)
    summary["subscriptions"] = len(sub_rows)
    # Build a plan-lookup so invoice + overage logic can use plan metadata
    plans_by_id = {p["id"]: p for p in plan_rows}

    # ---- seat_allocations (1 per org until count is reached) ----
    sa_target = min(counts["seat_allocations"], len(org_ids))
    sa_rows: list[dict[str, Any]] = []
    for i, oid in enumerate(org_ids[:sa_target]):
        sub = org_to_sub.get(oid)
        plan = plans_by_id[sub["plan_id"]] if sub else plans_by_id[plan_ids[0]]
        allocated = plan["included_seats"] + rng.randint(-2, 10)
        used = max(0, allocated - rng.randint(0, max(1, allocated // 2)))
        sa_rows.append(
            {
                "id": i + 1,
                "organization_id": oid,
                "allocated_seats": max(0, allocated),
                "used_seats": used,
            }
        )
    _bulk_insert(session, SeatAllocation, sa_rows)
    summary["seat_allocations"] = len(sa_rows)

    # ---- invoices ----
    sub_id_list = [s["id"] for s in sub_rows]
    org_id_list = [s["organization_id"] for s in sub_rows]
    invoice_rows: list[dict[str, Any]] = []
    for i in range(counts["invoices"]):
        inv_id = i + 1
        # Round-robin invoices across subscriptions so every sub gets some.
        if sub_id_list:
            idx = i % len(sub_id_list)
            sub_id = sub_id_list[idx]
            org_id = org_id_list[idx]
        else:
            sub_id = None
            org_id = rng.choice(org_ids)
        status = rng.choices(_INVOICE_STATUSES, weights=_INVOICE_STATUS_WEIGHTS)[0]
        issued = now - timedelta(days=rng.randint(0, 540))
        due = issued + timedelta(days=14)
        paid = issued + timedelta(days=rng.randint(1, 30)) if status == "paid" else None
        amount = Decimal(str(rng.randint(29, 5000))).quantize(Decimal("0.01"))
        invoice_rows.append(
            {
                "id": inv_id,
                "organization_id": org_id,
                "subscription_id": sub_id,
                "invoice_number": f"INV-{inv_id:06d}",
                "status": status,
                "total_amount": amount,
                "currency": "USD",
                "issued_at": issued,
                "due_at": due,
                "paid_at": paid,
            }
        )
    _bulk_insert(session, Invoice, invoice_rows)
    summary["invoices"] = len(invoice_rows)

    # ---- invoice_items (round-robin across invoices) ----
    item_target = counts["invoice_items"]
    item_rows: list[dict[str, Any]] = []
    for i in range(item_target):
        inv = invoice_rows[i % len(invoice_rows)] if invoice_rows else None
        if inv is None:
            break
        template = rng.choice(_INVOICE_LINE_TEMPLATES)
        plan_label = (
            plans_by_id.get(org_to_sub.get(inv["organization_id"], {}).get("plan_id", 0), {})
            .get("name", "Starter")
        )
        n = rng.randint(1, 20)
        rate = Decimal("0.30")
        desc = template.format(plan=plan_label, n=n, rate=f"${rate:.2f}/1k")
        amount = Decimal(str(rng.randint(5, 800))).quantize(Decimal("0.01"))
        item_rows.append(
            {
                "id": i + 1,
                "invoice_id": inv["id"],
                "description": desc,
                "amount": amount,
                "quantity": rng.randint(1, 10),
            }
        )
    _bulk_insert(session, InvoiceItem, item_rows)
    summary["invoice_items"] = len(item_rows)

    # ---- usage_events ----
    event_rows: list[dict[str, Any]] = []
    for i in range(counts["usage_events"]):
        oid = rng.choice(org_ids)
        et = rng.choices(_USAGE_EVENT_TYPES, weights=_USAGE_EVENT_WEIGHTS)[0]
        when = now - timedelta(
            days=rng.randint(0, 60),
            seconds=rng.randint(0, 86_400),
        )
        qty = rng.randint(1, 50) if et == "api_call" else rng.randint(1, 3)
        event_rows.append(
            {
                "id": i + 1,
                "organization_id": oid,
                "event_type": et,
                "quantity": qty,
                "occurred_at": when,
            }
        )
    _bulk_insert(session, UsageEvent, event_rows)
    summary["usage_events"] = len(event_rows)

    # ---- api_usage_daily (unique on (org_id, date)) ----
    daily_target = counts["api_usage_daily"]
    days_per_org = max(1, daily_target // max(1, len(org_ids)))
    daily_rows: list[dict[str, Any]] = []
    today = now.date()
    for o_idx, oid in enumerate(org_ids):
        for d in range(days_per_org):
            if len(daily_rows) >= daily_target:
                break
            day = today - timedelta(days=d)
            calls = rng.randint(100, 50_000)
            failed = int(calls * rng.uniform(0.0, 0.05))
            daily_rows.append(
                {
                    "id": len(daily_rows) + 1,
                    "organization_id": oid,
                    "date": day,
                    "api_calls": calls,
                    "successful_calls": calls - failed,
                    "failed_calls": failed,
                }
            )
        if len(daily_rows) >= daily_target:
            break
    _bulk_insert(session, ApiUsageDaily, daily_rows)
    summary["api_usage_daily"] = len(daily_rows)

    # ---- overage_charges (only on a subset of invoices) ----
    overage_target = min(counts["overage_charges"], len(invoice_rows))
    overage_rows: list[dict[str, Any]] = []
    for i in range(overage_target):
        inv = invoice_rows[i % len(invoice_rows)]
        usage = rng.randint(1_000, 200_000)
        rate = Decimal("0.30")
        charge = (Decimal(usage) * rate / Decimal("1000")).quantize(Decimal("0.01"))
        overage_rows.append(
            {
                "id": i + 1,
                "invoice_id": inv["id"],
                "organization_id": inv["organization_id"],
                "usage_amount": usage,
                "charge_amount": charge,
                "reason": rng.choice(_OVERAGE_REASONS),
            }
        )
    _bulk_insert(session, OverageCharge, overage_rows)
    summary["overage_charges"] = len(overage_rows)


# ---------------------------------------------------------------------------
# Commerce / orders seeding (Phase B2)
# ---------------------------------------------------------------------------

# Static reference data
_TOP_LEVEL_CATEGORIES = [
    "Electronics",
    "Clothing",
    "Books",
    "Home & Garden",
    "Sports & Outdoors",
    "Beauty",
    "Toys",
    "Office",
]

_SUB_CATEGORY_TEMPLATES: list[tuple[str, list[str]]] = [
    ("Electronics", ["Headphones", "Smartphones", "Laptops", "Cameras", "Wearables"]),
    ("Clothing", ["T-Shirts", "Jackets", "Shoes", "Accessories"]),
    ("Books", ["Fiction", "Non-fiction", "Technical"]),
    ("Home & Garden", ["Kitchen", "Bedding", "Outdoor Furniture"]),
    ("Sports & Outdoors", ["Cycling", "Hiking", "Fitness"]),
    ("Beauty", ["Skincare", "Fragrance"]),
    ("Toys", ["Educational", "Action Figures"]),
    ("Office", ["Stationery", "Furniture"]),
]

_PRODUCT_NAME_TEMPLATES = [
    "{adj} {noun}",
    "{noun} {model}",
    "{brand} {noun} {model}",
]
_PRODUCT_ADJECTIVES = [
    "Wireless", "Smart", "Cotton", "Leather", "Premium", "Compact", "Ultra-light",
    "Eco-friendly", "Heavy-duty", "Adjustable", "Foldable", "Waterproof",
]
_PRODUCT_NOUNS = [
    "Headphones", "Backpack", "Tablet", "Mug", "Sneakers", "Notebook", "Lamp",
    "Bottle", "Watch", "Charger", "Keyboard", "Mouse", "Bag", "Jacket",
    "Shirt", "Speaker", "Camera", "Stand", "Cable", "Pen",
]
_PRODUCT_BRANDS = [
    "Acme", "Nova", "Vertex", "Apex", "Lumen", "Aero", "Polaris",
    "Halo", "Onyx", "Crest", "Pulse", "Quest",
]
_PRODUCT_MODELS = ["X1", "X2", "Pro", "Lite", "Mini", "Max", "S", "S+", "2024", "EE"]

_ATTRIBUTE_TEMPLATES_BY_CATEGORY_KEYWORD: dict[str, list[tuple[str, list[str]]]] = {
    "Electronics": [
        ("color", ["black", "silver", "white", "blue", "red"]),
        ("battery_hours", ["10", "20", "30", "40"]),
        ("warranty_months", ["12", "24", "36"]),
    ],
    "Clothing": [
        ("size", ["XS", "S", "M", "L", "XL"]),
        ("material", ["cotton", "polyester", "wool", "linen"]),
        ("color", ["black", "navy", "olive", "white", "gray"]),
    ],
    "Books": [
        ("language", ["english", "spanish", "french", "german"]),
        ("pages", ["120", "240", "360", "480"]),
        ("format", ["paperback", "hardcover", "ebook"]),
    ],
}
_GENERIC_ATTRIBUTES = [
    ("weight_kg", ["0.2", "0.5", "1.0", "2.0", "5.0"]),
    ("country_of_origin", ["USA", "Germany", "Japan", "Vietnam", "Italy"]),
]

_WAREHOUSE_NAMES = [
    ("West Distribution Center", "Reno", "USA"),
    ("Central Hub", "Kansas City", "USA"),
    ("East Coast Depot", "Edison", "USA"),
    ("UK Fulfilment", "Manchester", "UK"),
    ("EU Hub", "Rotterdam", "Netherlands"),
    ("APAC Hub", "Singapore", "Singapore"),
    ("ANZ Hub", "Sydney", "Australia"),
    ("South America Depot", "São Paulo", "Brazil"),
    ("Mexico Hub", "Monterrey", "Mexico"),
    ("Middle East Hub", "Dubai", "UAE"),
    ("Nordics DC", "Stockholm", "Sweden"),
    ("Iberia DC", "Madrid", "Spain"),
    ("DACH DC", "Munich", "Germany"),
    ("France DC", "Lyon", "France"),
    ("Eastern Europe DC", "Warsaw", "Poland"),
    ("Africa Hub", "Johannesburg", "South Africa"),
    ("Korea DC", "Incheon", "South Korea"),
    ("Japan DC", "Osaka", "Japan"),
    ("India DC", "Mumbai", "India"),
    ("Canada DC", "Toronto", "Canada"),
    ("West Canada DC", "Vancouver", "Canada"),
    ("Texas DC", "Dallas", "USA"),
    ("Florida DC", "Miami", "USA"),
    ("California DC", "Los Angeles", "USA"),
    ("Pacific NW DC", "Seattle", "USA"),
    ("Midwest DC", "Chicago", "USA"),
    ("Southeast DC", "Atlanta", "USA"),
    ("Northeast DC", "Boston", "USA"),
    ("Hawaii DC", "Honolulu", "USA"),
    ("Alaska DC", "Anchorage", "USA"),
    ("Ireland DC", "Dublin", "Ireland"),
    ("Switzerland DC", "Zurich", "Switzerland"),
    ("Belgium DC", "Brussels", "Belgium"),
    ("Austria DC", "Vienna", "Austria"),
    ("Denmark DC", "Copenhagen", "Denmark"),
    ("Norway DC", "Oslo", "Norway"),
    ("Finland DC", "Helsinki", "Finland"),
    ("Iceland DC", "Reykjavik", "Iceland"),
    ("Portugal DC", "Lisbon", "Portugal"),
    ("Italy DC", "Milan", "Italy"),
    ("Greece DC", "Athens", "Greece"),
    ("Turkey DC", "Istanbul", "Turkey"),
    ("Egypt DC", "Cairo", "Egypt"),
    ("Kenya DC", "Nairobi", "Kenya"),
    ("Nigeria DC", "Lagos", "Nigeria"),
    ("Argentina DC", "Buenos Aires", "Argentina"),
    ("Chile DC", "Santiago", "Chile"),
    ("Colombia DC", "Bogota", "Colombia"),
    ("Peru DC", "Lima", "Peru"),
    ("Thailand DC", "Bangkok", "Thailand"),
]

_ORDER_STATUSES = ["placed", "processing", "shipped", "delivered", "cancelled", "returned"]
_ORDER_STATUS_WEIGHTS = [0.15, 0.10, 0.25, 0.35, 0.10, 0.05]
_SHIPMENT_STATUSES = ["pending", "in_transit", "delivered", "exception", "lost"]
_SHIPMENT_STATUS_WEIGHTS = [0.10, 0.30, 0.55, 0.04, 0.01]
_CARRIERS = ["UPS", "FedEx", "DHL", "USPS", "Royal Mail"]
_RETURN_STATUSES = ["requested", "approved", "rejected", "completed"]
# Tilted toward approved/completed so a healthy share supports refunds.
_RETURN_STATUS_WEIGHTS = [0.10, 0.40, 0.05, 0.45]
_RETURN_REASONS = [
    "Wrong size",
    "Damaged on arrival",
    "Not as described",
    "No longer needed",
    "Better price elsewhere",
    "Wrong item shipped",
    "Quality issue",
]
_REFUND_STATUSES_COMMERCE = ["pending", "approved", "completed", "rejected"]
_REFUND_STATUS_WEIGHTS_COMMERCE = [0.25, 0.20, 0.50, 0.05]


def _rand_tracking(rng: random.Random) -> str:
    return "TRK-" + "".join(
        rng.choices(string.ascii_uppercase + string.digits, k=10)
    )


def _seed_commerce(
    *,
    session: Session,
    counts: dict[str, int],
    customer_ids: list[int],
    rng: random.Random,
    fake: Faker,
    summary: dict[str, int],
) -> None:
    """Seed the commerce/orders tables.

    Runs after airline + SaaS seeding. Only adds new RNG draws, so all
    prior data remains unchanged.
    """
    # ---- product_categories ----
    cat_target = counts["product_categories"]
    cat_rows: list[dict[str, Any]] = []
    # First, the top-level categories (no parent).
    top_level_ids: dict[str, int] = {}
    for i, name in enumerate(_TOP_LEVEL_CATEGORIES[: min(len(_TOP_LEVEL_CATEGORIES), cat_target)]):
        cid = i + 1
        top_level_ids[name] = cid
        cat_rows.append({"id": cid, "name": name, "parent_id": None})
    # Then subcategories under known parents.
    sub_specs = []
    for parent_name, subs in _SUB_CATEGORY_TEMPLATES:
        if parent_name not in top_level_ids:
            continue
        for s in subs:
            sub_specs.append((parent_name, s))
    rng.shuffle(sub_specs)
    for parent_name, sub_name in sub_specs:
        if len(cat_rows) >= cat_target:
            break
        cat_rows.append(
            {
                "id": len(cat_rows) + 1,
                "name": f"{parent_name} / {sub_name}",
                "parent_id": top_level_ids[parent_name],
            }
        )
    # Top up with synthetic categories if needed (for medium/large).
    while len(cat_rows) < cat_target:
        parent_name = rng.choice(list(top_level_ids))
        cat_rows.append(
            {
                "id": len(cat_rows) + 1,
                "name": f"{parent_name} / {fake.unique.word().title()}",
                "parent_id": top_level_ids[parent_name],
            }
        )
    _bulk_insert(session, ProductCategory, cat_rows)
    cat_ids = [c["id"] for c in cat_rows]
    summary["product_categories"] = len(cat_rows)

    # ---- products ----
    product_rows: list[dict[str, Any]] = []
    cat_name_by_id = {c["id"]: c["name"] for c in cat_rows}
    for i in range(counts["products"]):
        pid = i + 1
        category_id = rng.choice(cat_ids)
        adj = rng.choice(_PRODUCT_ADJECTIVES)
        noun = rng.choice(_PRODUCT_NOUNS)
        brand = rng.choice(_PRODUCT_BRANDS)
        model = rng.choice(_PRODUCT_MODELS)
        # Pick a template
        tpl = rng.choice(_PRODUCT_NAME_TEMPLATES)
        name = tpl.format(adj=adj, noun=noun, brand=brand, model=model)
        product_rows.append(
            {
                "id": pid,
                "sku": f"SKU-{pid:06d}",
                "name": name,
                "category_id": category_id,
                "description": f"{name} — high-quality {noun.lower()} from {brand}.",
                "brand": brand,
                "is_active": rng.random() < 0.95,
            }
        )
    _bulk_insert(session, Product, product_rows)
    product_ids = [p["id"] for p in product_rows]
    summary["products"] = len(product_rows)

    # ---- product_attributes ----
    attr_target = counts["product_attributes"]
    attr_rows: list[dict[str, Any]] = []
    for i in range(attr_target):
        product = product_rows[i % len(product_rows)]
        cat_name = cat_name_by_id[product["category_id"]]
        attrs_pool: list[tuple[str, list[str]]] = []
        for key, attrs in _ATTRIBUTE_TEMPLATES_BY_CATEGORY_KEYWORD.items():
            if key.lower() in cat_name.lower():
                attrs_pool.extend(attrs)
        if not attrs_pool:
            attrs_pool = list(_GENERIC_ATTRIBUTES)
        name, values = rng.choice(attrs_pool)
        attr_rows.append(
            {
                "id": i + 1,
                "product_id": product["id"],
                "attribute_name": name,
                "attribute_value": rng.choice(values),
            }
        )
    _bulk_insert(session, ProductAttribute, attr_rows)
    summary["product_attributes"] = len(attr_rows)

    # ---- product_prices ----
    price_target = counts["product_prices"]
    price_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for i in range(price_target):
        product = product_rows[i % len(product_rows)]
        # First price for each product is current (valid_to=None); subsequent are historical.
        is_first = i < len(product_rows)
        valid_from = now - timedelta(days=rng.randint(30, 720))
        valid_to: Optional[datetime] = None
        if not is_first:
            valid_to = valid_from + timedelta(days=rng.randint(30, 300))
        # Price tied loosely to category
        cat_name = cat_name_by_id[product["category_id"]]
        if "Electronics" in cat_name:
            base = rng.randint(60, 1500)
        elif "Clothing" in cat_name:
            base = rng.randint(15, 250)
        elif "Books" in cat_name:
            base = rng.randint(8, 60)
        elif "Home" in cat_name:
            base = rng.randint(20, 800)
        else:
            base = rng.randint(10, 500)
        price = Decimal(str(base)).quantize(Decimal("0.01"))
        price_rows.append(
            {
                "id": i + 1,
                "product_id": product["id"],
                "currency": "USD",
                "price": price,
                "valid_from": valid_from,
                "valid_to": valid_to,
            }
        )
    _bulk_insert(session, ProductPrice, price_rows)
    # Index current price (no valid_to) per product for order-item lookups.
    price_by_product: dict[int, Decimal] = {}
    for row in price_rows:
        if row["valid_to"] is None:
            price_by_product[row["product_id"]] = row["price"]
    summary["product_prices"] = len(price_rows)

    # ---- warehouses ----
    wh_count = min(counts["warehouses"], len(_WAREHOUSE_NAMES))
    wh_rows: list[dict[str, Any]] = []
    for i in range(wh_count):
        name, city, country = _WAREHOUSE_NAMES[i]
        wh_rows.append(
            {"id": i + 1, "name": name, "city": city, "country": country}
        )
    # Top up with synthetic warehouses if needed.
    while len(wh_rows) < counts["warehouses"]:
        wid = len(wh_rows) + 1
        wh_rows.append(
            {
                "id": wid,
                "name": f"Auxiliary DC {wid}",
                "city": fake.city(),
                "country": fake.country(),
            }
        )
    _bulk_insert(session, Warehouse, wh_rows)
    wh_ids = [w["id"] for w in wh_rows]
    summary["warehouses"] = len(wh_rows)

    # ---- product_inventory (unique (product, warehouse)) ----
    inv_target = counts["product_inventory"]
    inv_rows: list[dict[str, Any]] = []
    inv_id = 1
    # Round-robin across products × warehouses so we honour the unique constraint.
    pairs_needed = inv_target
    p_idx = 0
    while pairs_needed > 0:
        product = product_rows[p_idx % len(product_rows)]
        for wid in wh_ids:
            if pairs_needed <= 0:
                break
            inv_rows.append(
                {
                    "id": inv_id,
                    "product_id": product["id"],
                    "warehouse_id": wid,
                    "quantity_available": rng.randint(0, 250),
                }
            )
            inv_id += 1
            pairs_needed -= 1
        p_idx += 1
        if p_idx > len(product_rows) * len(wh_ids):
            break  # safety: exhausted all pairs
    _bulk_insert(session, ProductInventory, inv_rows)
    summary["product_inventory"] = len(inv_rows)

    # ---- commerce_orders ----
    order_rows: list[dict[str, Any]] = []
    for i in range(counts["commerce_orders"]):
        oid = i + 1
        cust_id = customer_ids[rng.randrange(len(customer_ids))]
        status = rng.choices(_ORDER_STATUSES, weights=_ORDER_STATUS_WEIGHTS)[0]
        # Order total is set after we generate items; placeholder for now.
        order_rows.append(
            {
                "id": oid,
                "order_number": f"ORD-{oid:06d}",
                "customer_id": cust_id,
                "status": status,
                "total_amount": Decimal("0.00"),
                "currency": "USD",
            }
        )
    summary["commerce_orders"] = len(order_rows)

    # ---- commerce_order_items ----
    item_target = counts["commerce_order_items"]
    item_rows: list[dict[str, Any]] = []
    totals_by_order: dict[int, Decimal] = {row["id"]: Decimal("0.00") for row in order_rows}
    for i in range(item_target):
        order = order_rows[i % len(order_rows)]
        product = product_rows[rng.randrange(len(product_rows))]
        qty = rng.randint(1, 5)
        unit_price = price_by_product.get(product["id"], Decimal("25.00"))
        totals_by_order[order["id"]] += (unit_price * qty).quantize(Decimal("0.01"))
        item_rows.append(
            {
                "id": i + 1,
                "order_id": order["id"],
                "product_id": product["id"],
                "quantity": qty,
                "unit_price": unit_price,
            }
        )
    # Apply computed totals back to order rows before insert.
    for row in order_rows:
        row["total_amount"] = totals_by_order[row["id"]]
    _bulk_insert(session, CommerceOrder, order_rows)
    _bulk_insert(session, CommerceOrderItem, item_rows)
    summary["commerce_order_items"] = len(item_rows)

    # ---- shipments (a shipment can exist for any order, with a status that
    # matches the order's lifecycle: placed→pending, cancelled→exception,
    # delivered→delivered, etc.) Round-robin across the order list. ----
    ship_target = counts["shipments"]
    shipment_rows: list[dict[str, Any]] = []
    used_tracking: set[str] = set()
    for i in range(ship_target):
        order = order_rows[i % len(order_rows)]
        # Default to the weighted shipment-status distribution; nudge by order status.
        status = rng.choices(_SHIPMENT_STATUSES, weights=_SHIPMENT_STATUS_WEIGHTS)[0]
        if order["status"] == "delivered":
            status = "delivered"
        elif order["status"] == "cancelled":
            status = rng.choice(["pending", "exception"])
        elif order["status"] == "placed":
            status = "pending"
        # Tracking numbers are unique; resolve any collision deterministically.
        while True:
            tn = _rand_tracking(rng)
            if tn not in used_tracking:
                used_tracking.add(tn)
                break
        est = now + timedelta(days=rng.randint(-30, 14))
        shipment_rows.append(
            {
                "id": i + 1,
                "order_id": order["id"],
                "tracking_number": tn,
                "status": status,
                "carrier": rng.choice(_CARRIERS),
                "estimated_delivery": est,
            }
        )
    _bulk_insert(session, Shipment, shipment_rows)
    summary["shipments"] = len(shipment_rows)

    # ---- commerce_returns (only on delivered/returned orders) ----
    returnable = [o for o in order_rows if o["status"] in ("delivered", "returned")]
    rng.shuffle(returnable)
    return_target = min(counts["commerce_returns"], len(returnable))
    return_rows: list[dict[str, Any]] = []
    for i in range(return_target):
        order = returnable[i]
        return_rows.append(
            {
                "id": i + 1,
                "order_id": order["id"],
                "status": rng.choices(_RETURN_STATUSES, weights=_RETURN_STATUS_WEIGHTS)[0],
                "reason": rng.choice(_RETURN_REASONS),
            }
        )
    _bulk_insert(session, CommerceReturn, return_rows)
    summary["commerce_returns"] = len(return_rows)

    # ---- commerce_refunds (only on approved/completed returns) ----
    refundable = [
        r for r in return_rows if r["status"] in ("approved", "completed")
    ]
    rng.shuffle(refundable)
    refund_target = min(counts["commerce_refunds"], len(refundable))
    refund_rows: list[dict[str, Any]] = []
    today = now.date()
    for i in range(refund_target):
        ret = refundable[i]
        order = next((o for o in order_rows if o["id"] == ret["order_id"]), None)
        order_total = order["total_amount"] if order else Decimal("50.00")
        ratio = Decimal(str(round(rng.uniform(0.4, 1.0), 2)))
        amount = (order_total * ratio).quantize(Decimal("0.01"))
        refund_rows.append(
            {
                "id": i + 1,
                "return_id": ret["id"],
                "refund_status": rng.choices(
                    _REFUND_STATUSES_COMMERCE, weights=_REFUND_STATUS_WEIGHTS_COMMERCE
                )[0],
                "refund_amount": amount,
                "expected_resolution_date": today + timedelta(days=rng.randint(-20, 30)),
            }
        )
    _bulk_insert(session, CommerceRefund, refund_rows)
    summary["commerce_refunds"] = len(refund_rows)


# ---------------------------------------------------------------------------
# Phase 6B-2: textual knowledge seeding
# ---------------------------------------------------------------------------


# 5 domains × 10 policy types = 50 policy combos. Each has:
# - a domain-appropriate body opener
# - a list of clause templates that read naturally for that domain
_POLICY_CATALOG: list[dict[str, Any]] = [
    # ---- airline ----
    {
        "domain": "airline",
        "policy_type": "refund_policy",
        "title": "Airline Refund Policy",
        "body": (
            "Refunds are issued when the airline cancels a flight, when a "
            "schedule change exceeds two hours, or under documented medical "
            "emergencies. Refundable fare classes also qualify for full or "
            "partial refunds depending on time of cancellation. Processing "
            "windows: 5–7 business days to original payment method; 1–2 "
            "billing cycles to appear on a credit card statement."
        ),
        "clauses": [
            ("eligibility", "Eligibility", "Refunds are eligible when the airline cancels a flight, when a schedule change exceeds 2 hours, or under refundable fare conditions.", "high", "all fares", None),
            ("non_refundable", "Non-refundable fares", "Basic, Saver, and Promo fares are non-refundable but may be eligible for travel credit valid for 12 months from issue.", "normal", "basic/saver/promo", None),
            ("medical_emergency", "Medical emergency exceptions", "Medical emergencies require a physician's note dated within 30 days. Approval is granted case by case and processed within 10 business days.", "high", "all fares", "Self-attested illness without documentation"),
            ("processing_time", "Processing time", "Refunds take 5-7 business days for credit cards, 10-14 days for bank transfers. Travel credit is available within 24 hours.", "normal", "all fares", None),
            ("partial_refund", "Partial refunds", "Partial refunds apply when only some segments of an itinerary are cancelled; unused segments are refunded at face value minus a processing fee.", "normal", "multi-segment itineraries", None),
            ("currency", "Refund currency", "Refunds are issued in the original currency of purchase. Foreign exchange variation between purchase and refund dates is not compensated.", "normal", "all fares", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "baggage_policy",
        "title": "Checked Baggage Policy",
        "body": (
            "Checked baggage allowances vary by route type and cabin class. "
            "Economy international: 1 bag up to 23kg. Business international: "
            "2 bags up to 32kg each. Domestic flights: 1 bag at half the "
            "international allowance for the cabin class. Excess baggage is "
            "billed at the rate published at time of booking."
        ),
        "clauses": [
            ("economy_intl", "Economy international", "Economy passengers on international flights may check one bag up to 23kg and one cabin bag up to 7kg.", "normal", "economy/international", None),
            ("business_intl", "Business international", "Business class passengers on international flights may check two bags up to 32kg each, plus 12kg cabin bag.", "normal", "business/international", None),
            ("domestic", "Domestic allowances", "Domestic flights allow one checked bag up to 18kg (economy) or 25kg (business). Cabin bag rules match international.", "normal", "domestic", None),
            ("excess", "Excess baggage fees", "Excess baggage is billed per kilogram at airport rates which can be 3x the pre-paid online rate.", "high", "all fares", None),
            ("lost_or_damaged", "Lost or damaged baggage", "Lost or damaged baggage must be reported at the airport baggage desk within 7 days of arrival.", "high", "all fares", "Personal items not in the checked bag"),
            ("sporting", "Sporting equipment", "Sporting equipment (skis, bikes, golf clubs) requires pre-booking 48 hours before departure and may incur a fixed handling fee.", "normal", "all fares", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "cancellation_policy",
        "title": "Cancellation Policy",
        "body": (
            "Cancellations follow the rules of the booking class. Voluntary "
            "cancellations forfeit non-refundable portions; involuntary "
            "cancellations (caused by the airline) entitle the passenger to a "
            "full refund or free rebooking on the next available flight."
        ),
        "clauses": [
            ("voluntary", "Voluntary cancellations", "Voluntary cancellations forfeit non-refundable portions and incur a cancellation fee per fare class.", "high", "all fares", None),
            ("involuntary", "Involuntary cancellations", "If the airline cancels a flight, passengers may choose a full refund or free rebooking on the next available flight.", "high", "all fares", None),
            ("missed_connection", "Missed connections", "If a delay causes a missed onward connection, the airline rebooks at no charge on the next available flight; meal vouchers issued for waits >4h.", "high", "multi-segment itineraries", "Connections booked separately"),
            ("rebooking_window", "Rebooking window", "Rebooking on involuntary cancellations is valid for travel within 12 months from the original date.", "normal", "all fares", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "subscription_policy",
        "title": "Loyalty Program Terms",
        "body": (
            "Loyalty members earn points based on fare class and distance "
            "flown. Tier qualification is annual and based on calendar-year "
            "activity. Tier benefits include priority check-in, lounge access, "
            "and free upgrades on a standby basis."
        ),
        "clauses": [
            ("earning", "Earning points", "Points are earned based on the published earning chart for fare class and distance flown.", "normal", "all members", None),
            ("redemption", "Redemption", "Points can be redeemed for flights, upgrades, and partner products through the loyalty portal.", "normal", "all members", None),
            ("expiry", "Expiry", "Points expire 24 months after the last qualifying activity. A single qualifying activity resets the expiry clock.", "high", "all members", None),
            ("tier_qualification", "Tier qualification", "Tier status is calculated annually based on calendar-year tier-qualifying segments and spend.", "normal", "all members", None),
            ("standby_upgrades", "Standby upgrades", "Free upgrades for elite tiers are available on a standby basis 24 hours before departure.", "normal", "gold/platinum", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "warranty_policy",
        "title": "Service Guarantee",
        "body": (
            "We guarantee on-time departure for at least 80% of flights on a "
            "rolling 90-day basis. Customers affected by delays of >3 hours "
            "receive a travel voucher valid for 6 months."
        ),
        "clauses": [
            ("delay_compensation", "Delay compensation", "Delays of 3+ hours qualify for a $100 voucher; 6+ hours qualify for $250 and meal credit.", "high", "all fares", "Weather, ATC, force majeure"),
            ("on_time_definition", "On-time definition", "On-time is defined as departure within 15 minutes of scheduled time per industry standard.", "normal", "all fares", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "escalation_policy",
        "title": "Airline Escalation Policy",
        "body": (
            "Customer issues are escalated based on severity. Routine inquiries "
            "are resolved by frontline support within 48 hours. Material "
            "complaints route to supervisors with same-business-day response."
        ),
        "clauses": [
            ("severity_definitions", "Severity definitions", "Severity 1: safety, regulatory, or media impact. Severity 2: customer financial impact >$500. Severity 3: standard inquiry.", "high", "all tickets", None),
            ("sla_response", "Response SLA", "Sev1: 1 hour. Sev2: 4 hours. Sev3: 48 hours.", "high", "all tickets", "Outside business hours adds 12h to Sev3"),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "payment_policy",
        "title": "Payment Terms",
        "body": (
            "All bookings require full payment at the time of reservation. We "
            "accept major credit cards, debit cards, and select digital wallets."
        ),
        "clauses": [
            ("chargebacks", "Chargebacks", "Disputed chargebacks invalidate the booking. The customer is responsible for any rebooking fees if the dispute is later reversed.", "high", "all fares", None),
            ("split_payments", "Split payments", "A single booking can be split across at most two payment methods.", "normal", "all fares", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "privacy_policy",
        "title": "Privacy Policy (Airline)",
        "body": (
            "We collect personal data necessary to process bookings, including "
            "passenger name, contact details, and travel document numbers. "
            "Data is retained per regulatory requirements and shared with "
            "authorities as required by destination country."
        ),
        "clauses": [
            ("retention", "Retention period", "Booking data is retained for 7 years for tax and regulatory purposes.", "high", "all customers", None),
            ("third_party", "Third-party sharing", "Passenger data is shared with destination authorities as required by law (e.g. APIS).", "high", "international travel", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "overage_policy",
        "title": "Overage / Excess Charges (Airline)",
        "body": (
            "Excess baggage, seat upgrades at check-in, and unaccompanied minor "
            "fees fall under our overage charges schedule, published in the "
            "fee tariff."
        ),
        "clauses": [
            ("excess_bag", "Excess baggage charges", "Excess baggage is billed per kg at the published airport rate.", "normal", "all fares", None),
        ],
    },
    {
        "domain": "airline",
        "policy_type": "return_policy",
        "title": "Travel Credit Use Policy",
        "body": (
            "Travel credits issued from cancellations or vouchers are valid "
            "for 12 months from issue and can be applied to fares for the "
            "original passenger or designated family members."
        ),
        "clauses": [
            ("transfer", "Transferability", "Travel credits can be transferred to family members upon written request and presentation of ID.", "normal", "all members", None),
        ],
    },
    # ---- commerce ----
    {
        "domain": "commerce",
        "policy_type": "return_policy",
        "title": "Commerce Standard Return Policy",
        "body": (
            "Most products may be returned within 30 days of delivery for a "
            "full refund. Items must be unused and in original packaging. "
            "Categories with hygiene, custom-made, or final-sale designations "
            "are non-returnable."
        ),
        "clauses": [
            ("window", "Return window", "Standard return window is 30 days from delivery; electronics 14 days; furniture 7 days.", "high", "all categories", None),
            ("opened_items", "Opened items", "Opened items in resaleable condition may be returned with a 15% restocking fee.", "normal", "non-hygiene categories", None),
            ("hygiene", "Hygiene exclusions", "Hygiene products (swimwear, undergarments, personal grooming) are non-returnable once seal is broken.", "high", "hygiene", None),
            ("damaged_packaging", "Damaged packaging", "Damaged packaging must be reported with photos within 48 hours of delivery to qualify for a free replacement.", "high", "all categories", None),
            ("missing_accessories", "Missing accessories", "Missing accessories (cables, adapters, manuals) must be reported within 7 days; we ship replacements at no cost.", "normal", "electronics/appliances", None),
            ("final_sale", "Final sale", "Items marked final sale on the product page are non-returnable.", "high", "marked items", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "refund_policy",
        "title": "Commerce Refund Policy",
        "body": (
            "Refunds are issued upon receipt of the returned item in "
            "acceptable condition. Refunds go to the original payment method "
            "within 5-7 business days. Shipping costs are non-refundable "
            "except in cases of carrier error or defective product."
        ),
        "clauses": [
            ("processing", "Refund processing", "Refunds are processed within 3 business days of return receipt and inspection.", "normal", "all returns", None),
            ("shipping_costs", "Shipping costs", "Shipping costs are non-refundable except when the product is defective or shipped in error.", "normal", "all returns", None),
            ("restocking_fee", "Restocking fee", "A 15% restocking fee applies to opened electronics and 20% to large furniture.", "high", "electronics/furniture", "Defective products"),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "warranty_policy",
        "title": "Commerce Product Warranty Policy",
        "body": (
            "Most products are covered by a manufacturer's warranty for 12-24 "
            "months. Extended warranties are available for purchase at "
            "checkout. Damage caused by misuse, modification, or normal wear "
            "is excluded."
        ),
        "clauses": [
            ("manufacturer", "Manufacturer warranty", "Manufacturer warranty covers defects in materials and workmanship for 12 months from purchase.", "high", "all electronics", None),
            ("exclusions", "Warranty exclusions", "Physical damage, water exposure, unauthorized modifications, and normal wear-and-tear are excluded from warranty coverage.", "high", "all warranties", None),
            ("extended", "Extended warranty", "Extended warranty plans can be purchased within 30 days of original purchase and extend coverage to 24 or 36 months.", "normal", "select categories", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "cancellation_policy",
        "title": "Order Cancellation Policy",
        "body": (
            "Orders can be cancelled at no charge until they enter the "
            "warehouse picking process. After picking, the order ships and "
            "must be processed as a return."
        ),
        "clauses": [
            ("pre_pick", "Before picking", "Orders can be cancelled free of charge while in 'placed' or 'processing' status.", "normal", "all orders", None),
            ("post_pick", "After picking", "Once an order is in 'shipped' status, it cannot be cancelled and must be returned upon delivery.", "high", "all orders", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "payment_policy",
        "title": "Commerce Payment Terms",
        "body": (
            "We accept all major credit cards, debit cards, and digital "
            "wallets. Charges are authorized at order placement and captured "
            "at shipment."
        ),
        "clauses": [
            ("authorization", "Authorization vs capture", "Cards are authorized at order placement and captured when the order ships.", "normal", "all orders", None),
            ("partial_capture", "Partial captures", "Partial shipments result in partial captures matching the value of the shipped items.", "normal", "split shipments", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "privacy_policy",
        "title": "Commerce Privacy Policy",
        "body": (
            "We collect order, payment, and shipping data necessary to "
            "fulfill purchases. Marketing communications require explicit "
            "opt-in."
        ),
        "clauses": [
            ("marketing", "Marketing communications", "Marketing emails require explicit opt-in. Unsubscribe links are included in every marketing email.", "normal", "all customers", None),
            ("data_sharing", "Data sharing with carriers", "Shipping addresses and contact info are shared with carriers to complete delivery.", "normal", "all orders", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "escalation_policy",
        "title": "Commerce Escalation Policy",
        "body": (
            "Order issues are escalated based on order value and customer "
            "tier. High-value orders and elite customers receive accelerated "
            "handling."
        ),
        "clauses": [
            ("high_value", "High-value orders", "Orders over $1,000 are routed to senior support with priority handling.", "high", "high-value orders", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "overage_policy",
        "title": "Commerce Overage / Surcharge Policy",
        "body": (
            "Oversized or heavy items may carry shipping surcharges disclosed "
            "at checkout. Surcharges are calculated by package weight and "
            "longest dimension and are itemised on the invoice."
        ),
        "clauses": [
            ("oversized", "Oversized shipping surcharge", "Items exceeding 50kg or 1.5m in any dimension carry a $50 oversized handling surcharge.", "normal", "large items", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "subscription_policy",
        "title": "Commerce Subscribe-and-Save Policy",
        "body": (
            "Subscribe-and-save orders ship on a recurring schedule with a "
            "10% discount applied to each recurring shipment."
        ),
        "clauses": [
            ("modification", "Modification window", "Subscription modifications must be made at least 48 hours before the next ship date.", "normal", "all subscriptions", None),
            ("cancel", "Cancellation", "Subscriptions can be cancelled at any time; no fees apply.", "normal", "all subscriptions", None),
        ],
    },
    {
        "domain": "commerce",
        "policy_type": "baggage_policy",
        "title": "Commerce 'Bag Check' (not applicable)",
        "body": (
            "Note: 'Baggage' applies to the airline domain. This entry is "
            "retained for cross-domain disambiguation testing and contains "
            "no actual commerce baggage rules."
        ),
        "clauses": [
            ("see_airline", "See airline baggage policy", "For baggage allowances, see the airline domain's baggage_policy document.", "normal", "n/a", None),
        ],
    },
    # ---- saas ----
    {
        "domain": "saas",
        "policy_type": "overage_policy",
        "title": "SaaS API Overage Policy",
        "body": (
            "API call overages above the included plan quota are billed at "
            "the plan's per-1000-call rate. A 5% grace period applies before "
            "billing begins; thereafter, overages are billed monthly on the "
            "next invoice."
        ),
        "clauses": [
            ("grace", "Grace period", "Each plan has a 5% over-quota grace period before overage billing begins.", "normal", "all plans", None),
            ("rates", "Overage rates", "Starter: $0.50/1k calls. Pro: $0.30/1k. Business: $0.20/1k. Enterprise: $0.10/1k.", "high", "all plans", None),
            ("hard_cap", "Hard cap", "Hard caps can be configured per organization; once reached, API calls return HTTP 429 until the next billing cycle.", "high", "all plans", "Enterprise tier with negotiated terms"),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "subscription_policy",
        "title": "SaaS Subscription Policy",
        "body": (
            "Subscriptions auto-renew monthly or annually depending on the "
            "selected billing cadence. Cancellation can be done from the "
            "account page or by contacting support."
        ),
        "clauses": [
            ("auto_renewal", "Auto-renewal", "Subscriptions auto-renew unless cancelled at least 24 hours before the next billing date.", "normal", "all plans", None),
            ("cancellation", "Cancellation", "Cancellation takes effect at the end of the current billing period; no refunds for the remaining period.", "high", "all plans", "Enterprise contracts with custom terms"),
            ("downgrade", "Downgrade limitations", "Downgrading to a lower plan may forfeit usage above the new plan's limits. Use the seat reconciliation tool first.", "high", "all plans", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "payment_policy",
        "title": "SaaS Payment Terms",
        "body": (
            "Invoices are due on the 1st of each month. Failed payments "
            "trigger an automated retry sequence followed by suspension."
        ),
        "clauses": [
            ("invoice_due", "Invoice due dates", "Monthly invoices are due on issue date + 14 days. Annual contracts are due on issue date + 30 days.", "high", "all plans", None),
            ("failed_payment", "Failed payments", "Failed payments retry on days 1, 3, 5, 7. After 4 failures, the account is suspended until payment resolves.", "high", "all plans", None),
            ("disputes", "Invoice disputes", "Invoice disputes must be raised within 30 days of issue. Disputed amounts are held pending resolution.", "high", "all plans", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "privacy_policy",
        "title": "SaaS Privacy Policy",
        "body": (
            "Customer data is processed per the data processing addendum. "
            "We do not sell customer data and provide audit logs on request "
            "for enterprise customers."
        ),
        "clauses": [
            ("dpa", "Data processing addendum", "Enterprise customers receive a signed DPA on request; SCCs are included for EEA transfers.", "high", "enterprise", None),
            ("audit_logs", "Audit logs", "Audit logs are retained for 90 days on standard plans, 365 days on enterprise.", "normal", "all plans", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "escalation_policy",
        "title": "SaaS Incident Escalation",
        "body": (
            "Incidents are classified P1-P4. P1 is a full outage affecting "
            "all tenants; P4 is a documentation question."
        ),
        "clauses": [
            ("p1_definition", "P1 definition", "P1: full outage affecting all tenants OR a customer-impacting security incident. Pages on-call immediately.", "high", "all customers", None),
            ("sla", "Incident SLA", "P1: 15min ack, 1hr first update. P2: 1hr ack, 4hr update. P3: 1 business day. P4: 3 business days.", "high", "all customers", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "refund_policy",
        "title": "SaaS Credit / Refund Policy",
        "body": (
            "We do not offer cash refunds for subscriptions. Service credits "
            "are issued for SLA breaches per the published SLA agreement."
        ),
        "clauses": [
            ("credits", "Service credits", "Service credits are issued for SLA breaches at the rate of 10% of the monthly subscription per hour of downtime, capped at 50%.", "high", "all paid plans", "Promotional credits"),
            ("no_cash", "No cash refunds", "Subscription fees are non-refundable. Service credits are applied against the next invoice.", "high", "all paid plans", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "warranty_policy",
        "title": "SaaS Service Level Agreement",
        "body": (
            "We commit to 99.9% uptime measured monthly. SLA breaches result "
            "in service credits per the credit policy."
        ),
        "clauses": [
            ("uptime_target", "Uptime target", "Monthly uptime target is 99.9% (43 minutes of downtime per month maximum).", "high", "all paid plans", None),
            ("scheduled_maintenance", "Scheduled maintenance", "Scheduled maintenance windows are excluded from downtime calculations when announced 7+ days in advance.", "normal", "all customers", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "cancellation_policy",
        "title": "SaaS Cancellation Policy",
        "body": (
            "Customers can cancel via the account page or by contacting "
            "support. Annual plans paid up front are not refunded for the "
            "unused portion."
        ),
        "clauses": [
            ("end_of_period", "End-of-period cancellation", "Cancellation takes effect at the end of the current billing period.", "normal", "all plans", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "return_policy",
        "title": "SaaS Refund / Return (n/a)",
        "body": (
            "SaaS subscriptions are intangible products and are not 'returned'. "
            "Refund-like behaviour is handled via the SaaS refund policy."
        ),
        "clauses": [
            ("see_refund", "See refund policy", "For credits and refund-like remedies, see the SaaS refund policy document.", "normal", "all plans", None),
        ],
    },
    {
        "domain": "saas",
        "policy_type": "baggage_policy",
        "title": "SaaS 'Baggage' (cross-domain only)",
        "body": (
            "This entry exists to surface the cross-domain 'seat'/'baggage' "
            "vocabulary collision. SaaS does not have a baggage policy."
        ),
        "clauses": [
            ("seats_meaning", "SaaS seats vs airline seats", "In SaaS context, 'seats' refers to user seats on an organization's plan. For airline seats, see the airline baggage and seat-selection policies.", "high", "all plans", None),
        ],
    },
    # ---- support ----
    {
        "domain": "support",
        "policy_type": "escalation_policy",
        "title": "Support Escalation Policy",
        "body": (
            "Support tickets are triaged by priority: low/normal/high/urgent. "
            "Each priority has defined SLAs and escalation paths."
        ),
        "clauses": [
            ("priority_definitions", "Priority definitions", "Urgent: safety/operational impact. High: material customer financial impact. Normal: routine inquiry. Low: documentation/educational.", "high", "all tickets", None),
            ("sla", "Response SLAs", "Urgent: 1 hour. High: 4 hours. Normal: 24 hours. Low: 48 hours.", "high", "all tickets", "Outside business hours adds 12h to Normal/Low"),
            ("manual_review", "Manual review", "Refunds above $500 require manual review by a supervisor.", "high", "refund tickets", None),
            ("retention", "Retention escalation", "Tickets from VIP / corporate customers are auto-escalated to senior support.", "normal", "VIP / corporate", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "refund_policy",
        "title": "Support Refund Approval Process",
        "body": (
            "Refund requests flow through automated checks first; manual "
            "review applies to refunds above defined thresholds."
        ),
        "clauses": [
            ("auto_approve", "Auto-approval", "Refunds under $100 with documented eligibility are auto-approved within 24 hours.", "normal", "all tickets", None),
            ("manual_threshold", "Manual review threshold", "Refunds at or above $500 require supervisor approval and documentation.", "high", "all tickets", None),
            ("fraud_review", "Fraud review", "Refunds flagged by fraud signals route to the risk team for investigation before approval.", "high", "flagged tickets", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "privacy_policy",
        "title": "Support Privacy Policy",
        "body": (
            "Support communication is logged for quality and training. "
            "Customers may request access to or deletion of their support "
            "history per privacy regulations."
        ),
        "clauses": [
            ("logging", "Communication logging", "Email, chat, and call transcripts are logged for up to 24 months for quality and dispute resolution.", "normal", "all tickets", None),
            ("dsar", "Data subject access", "Customers may request a copy of their support records at any time; we respond within 30 days.", "high", "all customers", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "cancellation_policy",
        "title": "Support Case Closure Policy",
        "body": (
            "Tickets are closed after resolution and customer confirmation. "
            "Unconfirmed tickets auto-close after 14 days of inactivity."
        ),
        "clauses": [
            ("auto_close", "Auto-close window", "Tickets with no customer activity for 14 days auto-close. The customer can reopen within 30 days.", "normal", "all tickets", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "payment_policy",
        "title": "Support Goodwill Credits Policy",
        "body": (
            "Goodwill credits compensate for service failures and are issued "
            "at the discretion of support agents within published limits."
        ),
        "clauses": [
            ("limits", "Per-issue limits", "Frontline agents may issue credits up to $50 without supervisor approval. $50-$200 requires supervisor sign-off.", "high", "all agents", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "overage_policy",
        "title": "Support Goodwill Overage Tracking",
        "body": (
            "Goodwill credits exceeding the published per-customer cap trigger "
            "an alert to the retention team for review."
        ),
        "clauses": [
            ("cap", "Customer cap", "No more than $500 of goodwill credit per customer per rolling 12 months without retention team approval.", "high", "all agents", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "warranty_policy",
        "title": "Support Resolution Guarantee",
        "body": (
            "Our support guarantee: a first response within SLA and a clear "
            "path to resolution communicated by the second response."
        ),
        "clauses": [
            ("first_response", "First response", "First response time matches the priority SLA. Auto-responses do not count as a first response.", "high", "all tickets", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "return_policy",
        "title": "Support Re-open Policy",
        "body": (
            "Customers may re-open a resolved ticket within 30 days of "
            "closure. Reopened tickets keep the original ticket number."
        ),
        "clauses": [
            ("window", "Re-open window", "Re-open within 30 days; after that, a new ticket is created.", "normal", "all tickets", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "subscription_policy",
        "title": "Support Premium Tier Policy",
        "body": (
            "Premium support is a paid add-on that includes accelerated SLAs "
            "and a named technical account manager."
        ),
        "clauses": [
            ("sla_uplift", "SLA uplift", "Premium support cuts SLA response times by 50% across all priorities.", "high", "premium customers", None),
        ],
    },
    {
        "domain": "support",
        "policy_type": "baggage_policy",
        "title": "Support 'Baggage' (cross-domain disambiguation)",
        "body": (
            "Support agents handling baggage-related questions should route "
            "to the airline domain's baggage policy."
        ),
        "clauses": [
            ("route", "Route to airline", "Baggage questions on support tickets must reference the airline domain's baggage_policy document.", "normal", "support agents", None),
        ],
    },
    # ---- crm ----
    {
        "domain": "crm",
        "policy_type": "privacy_policy",
        "title": "Customer Data Privacy Policy",
        "body": (
            "Customer personal data is processed under the privacy notice. "
            "Customers have rights to access, correct, and delete their data "
            "subject to legal retention requirements."
        ),
        "clauses": [
            ("retention", "Retention", "Personal data is retained for 7 years after the last interaction for regulatory purposes.", "high", "all customers", "Legal hold or active dispute"),
            ("deletion_request", "Deletion requests", "Customers may request deletion; we respond within 30 days and confirm scope before action.", "high", "all customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "subscription_policy",
        "title": "Customer Marketing Preferences",
        "body": (
            "Customers can manage marketing preferences from the account "
            "page. Transactional emails are sent regardless of marketing "
            "preferences."
        ),
        "clauses": [
            ("opt_in", "Marketing opt-in", "Marketing communications require explicit opt-in. Opt-in status is recorded with timestamp and source.", "high", "all customers", None),
            ("transactional", "Transactional emails", "Transactional emails (orders, bookings, billing) are always sent and cannot be opted out of.", "normal", "all customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "escalation_policy",
        "title": "VIP / Risk Escalation",
        "body": (
            "VIP customers receive accelerated handling. Customers flagged "
            "for risk receive additional verification before changes."
        ),
        "clauses": [
            ("vip_handling", "VIP handling", "Customers tagged VIP receive senior support routing automatically.", "high", "VIP customers", None),
            ("risk_flag", "Risk flag verification", "Customers flagged for risk require ID verification before account changes.", "high", "flagged customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "payment_policy",
        "title": "Customer Payment Methods on File",
        "body": (
            "Payment methods stored on the customer profile are tokenized "
            "and PCI-compliant. Customers can add, update, or remove payment "
            "methods at any time."
        ),
        "clauses": [
            ("storage", "Tokenized storage", "Payment methods are stored as tokens; full card numbers are never retained.", "high", "all customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "refund_policy",
        "title": "Customer-Level Refund Eligibility",
        "body": (
            "Refund eligibility considers customer history. Repeated refund "
            "requests trigger a review by the retention team."
        ),
        "clauses": [
            ("history_check", "Refund history check", "Refunds beyond the 3rd in a rolling 12 months trigger a retention team review.", "high", "all customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "cancellation_policy",
        "title": "Customer Account Closure",
        "body": (
            "Account closure deletes personal data per the retention policy "
            "and cancels active subscriptions. Pending refunds are honored."
        ),
        "clauses": [
            ("data_handling", "Data on closure", "On account closure, identifying data is deleted within 30 days; transactional history is retained per regulatory retention.", "high", "all customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "return_policy",
        "title": "Customer-Wide Return Eligibility",
        "body": (
            "Customers with abnormally high return rates may be flagged for "
            "review by the fraud team."
        ),
        "clauses": [
            ("threshold", "Return rate threshold", "Return rates above 40% in any 90-day window trigger a fraud team review.", "high", "all customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "warranty_policy",
        "title": "Customer Service Warranty Tracking",
        "body": (
            "Customer-level warranty tracking records all extended warranties "
            "purchased and links them to the original product purchase."
        ),
        "clauses": [
            ("linkage", "Product linkage", "Extended warranties are linked to the original order_item_id for traceability.", "normal", "all customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "overage_policy",
        "title": "Customer-Level Overage Aggregation",
        "body": (
            "Where a customer maintains multiple SaaS organizations, overages "
            "are aggregated at the customer level for retention review."
        ),
        "clauses": [
            ("aggregation", "Cross-org aggregation", "Customers with overages exceeding $1,000 across organizations in a month trigger retention outreach.", "high", "multi-org customers", None),
        ],
    },
    {
        "domain": "crm",
        "policy_type": "baggage_policy",
        "title": "Customer 'Baggage' (cross-domain disambiguation)",
        "body": (
            "Cross-domain disambiguation note: customer-level baggage tags "
            "do not exist; route to the airline domain's baggage policy."
        ),
        "clauses": [
            ("route", "Route to airline", "Customer-level baggage questions must use the airline domain's baggage_policy.", "normal", "all customers", None),
        ],
    },
]


# Warranty templates per warranty_type
_WARRANTY_TEMPLATES = [
    {
        "warranty_type": "manufacturer_standard",
        "duration_months": 12,
        "body_template": (
            "Manufacturer's standard warranty covering defects in materials "
            "and workmanship for 12 months from date of purchase. Coverage "
            "includes free repair or replacement of defective parts. "
            "Customers must retain the original receipt and product packaging."
        ),
        "exclusions": "Physical damage, water exposure, unauthorized modifications, normal wear and tear, cosmetic damage.",
    },
    {
        "warranty_type": "manufacturer_extended",
        "duration_months": 24,
        "body_template": (
            "Extended manufacturer's warranty covering defects in materials "
            "and workmanship for 24 months. Includes free shipping for "
            "warranty claims within the United States."
        ),
        "exclusions": "Physical damage, water exposure, software issues, accessory failures, and consumable parts (batteries, fuses).",
    },
    {
        "warranty_type": "premium_extended",
        "duration_months": 36,
        "body_template": (
            "Premium extended warranty for 36 months including accidental "
            "damage protection. Covers one accidental damage incident per "
            "12-month period with a $50 service fee."
        ),
        "exclusions": "Loss or theft, intentional damage, software-related issues, modifications voiding the warranty.",
    },
    {
        "warranty_type": "limited_lifetime",
        "duration_months": 240,  # 20 years; treat as 'lifetime'
        "body_template": (
            "Limited lifetime warranty for the original purchaser. Covers "
            "structural defects in materials and workmanship for the useful "
            "life of the product, defined as 20 years."
        ),
        "exclusions": "Wear-and-tear, fading, fabric pilling, and damage from normal use.",
    },
    {
        "warranty_type": "no_warranty",
        "duration_months": 0,
        "body_template": (
            "This product is sold without warranty. Final sale. Refunds and "
            "returns follow the product return policy only."
        ),
        "exclusions": "All claims; product is sold as-is.",
    },
    {
        "warranty_type": "third_party",
        "duration_months": 12,
        "body_template": (
            "Third-party warranty administered by the manufacturer directly. "
            "Customers must contact the manufacturer using the warranty card "
            "shipped with the product."
        ),
        "exclusions": "Coverage subject to the third-party warranty terms; we do not handle claims directly.",
    },
]


# Return rule templates per category
_RETURN_RULE_TEMPLATES = [
    {
        "rule_name": "Standard 30-day return",
        "body": (
            "Items in this category may be returned within 30 days of "
            "delivery for a full refund. Items must be in original packaging "
            "and unused. Return shipping is the customer's responsibility "
            "unless the item is defective."
        ),
        "opened_item_allowed": True,
        "return_window_days": 30,
        "restocking_fee_percent": Decimal("0.00"),
        "exceptions": "Final sale items, promotional bundles.",
    },
    {
        "rule_name": "Electronics — 14-day return",
        "body": (
            "Electronics may be returned within 14 days of delivery. Opened "
            "items in resaleable condition are accepted with a 15% restocking "
            "fee. Items with broken seals, water damage, or evidence of "
            "modification are non-returnable."
        ),
        "opened_item_allowed": True,
        "return_window_days": 14,
        "restocking_fee_percent": Decimal("15.00"),
        "exceptions": "Broken seals, water damage, unauthorized modifications, missing serial number.",
    },
    {
        "rule_name": "Furniture — 7-day inspection",
        "body": (
            "Furniture must be inspected at delivery; visible damage must be "
            "noted on the delivery receipt. Returns accepted within 7 days, "
            "subject to a 20% restocking fee and customer-paid return shipping."
        ),
        "opened_item_allowed": True,
        "return_window_days": 7,
        "restocking_fee_percent": Decimal("20.00"),
        "exceptions": "Custom-made or made-to-order furniture (non-returnable).",
    },
    {
        "rule_name": "Hygiene — non-returnable once opened",
        "body": (
            "Hygiene products (swimwear, undergarments, personal grooming, "
            "earphones with in-ear tips) are non-returnable once the seal or "
            "packaging is broken. Unopened items in original packaging may be "
            "returned within 30 days."
        ),
        "opened_item_allowed": False,
        "return_window_days": 30,
        "restocking_fee_percent": Decimal("0.00"),
        "exceptions": "Items shipped damaged or incorrect.",
    },
    {
        "rule_name": "Damaged packaging — photo within 48 hours",
        "body": (
            "Items received with damaged packaging must be photographed and "
            "reported within 48 hours of delivery. Approved claims receive a "
            "free replacement or full refund."
        ),
        "opened_item_allowed": True,
        "return_window_days": 30,
        "restocking_fee_percent": Decimal("0.00"),
        "exceptions": "Damage reported after 48 hours.",
    },
    {
        "rule_name": "Missing accessories — 7-day report",
        "body": (
            "Missing accessories (cables, adapters, manuals, remotes) must be "
            "reported within 7 days of delivery. We ship replacements at no "
            "cost without requiring the main item to be returned."
        ),
        "opened_item_allowed": True,
        "return_window_days": 7,
        "restocking_fee_percent": Decimal("0.00"),
        "exceptions": "Accessories noted as 'not included' on the product page.",
    },
    {
        "rule_name": "Final sale — non-returnable",
        "body": (
            "Items marked as final sale on the product page are non-returnable "
            "regardless of condition. The final-sale designation is clearly "
            "indicated at the time of purchase."
        ),
        "opened_item_allowed": False,
        "return_window_days": 0,
        "restocking_fee_percent": Decimal("100.00"),
        "exceptions": "Items shipped damaged.",
    },
    {
        "rule_name": "Books — strict packaging",
        "body": (
            "Books and printed materials must be returned in the exact "
            "original packaging with no markings, dog-ears, or spine creases. "
            "Returns are processed within 14 days of receipt."
        ),
        "opened_item_allowed": True,
        "return_window_days": 30,
        "restocking_fee_percent": Decimal("0.00"),
        "exceptions": "Books with any visible use or damage.",
    },
]


# Customer note templates
_NOTE_TEMPLATES = [
    ("vip_handling", "VIP customer with multiple high-value bookings in the last 12 months. Always route to senior support. Preferences: window seat, vegetarian meal."),
    ("previous_complaint", "Customer raised a complaint about delayed refund processing on {related_type} {related_id}. Resolved with a $50 goodwill credit on 2026-02-14. Do not re-issue without supervisor approval."),
    ("special_handling", "Customer is a corporate travel arranger; bookings may include up to 12 passengers. Confirm passenger list against authorization list before changes."),
    ("retention_offer", "Retention offer extended on 2026-03-02: 20% off next 3 months. Valid until 2026-06-30. Mention only if customer raises pricing concern."),
    ("fraud_review", "Account flagged by fraud signals in March 2026 (multiple cards, mismatched billing/shipping). Cleared after manual review. Continue routine monitoring."),
    ("unresolved_issue", "Customer reports lost baggage on flight in February 2026; carrier compensation pending. Follow up monthly until closed."),
    ("language_preference", "Customer prefers Spanish-language communications. Account flag set 2026-01-15."),
    ("accessibility", "Customer requires wheelchair assistance; auto-add to every booking. Confirmed 2026-04-10."),
    ("payment_issue", "Two failed payments in March 2026 due to expired card. Customer updated payment method on 2026-04-01. Watch for repeated failures."),
    ("loyalty_status", "Loyalty status escalated to Platinum on 2026-01-30 due to corporate program. Apply Platinum benefits regardless of mileage thresholds."),
    ("escalation_history", "Customer escalated to supervisor twice in 2025 over flight changes. Notes from 2025-11-20 indicate frustration with rebooking policy."),
    ("subscription_downgrade", "Customer requested downgrade from Pro to Starter on 2026-02-28. Downgrade effective 2026-03-31. Watch for seat reconciliation issues."),
    ("commerce_dispute", "Customer disputed order {related_type} {related_id} as 'not received'; carrier proof of delivery on file. Future disputes require additional verification."),
    ("invoice_dispute", "Open invoice dispute on {related_id}: customer claims overage was already paid. Investigating with billing team."),
    ("courtesy_extension", "Granted a one-time courtesy extension on 2026-02-22 for return window from 30 to 45 days. Do not repeat without supervisor approval."),
    ("manual_review_required", "Refunds above $500 on this account require manual review until 2026-12-31."),
    ("legal_hold", "Account under legal hold; do not modify, delete, or merge. Contact legal@democorp.example before any action."),
    ("internal_test_account", "Internal test account used by engineering. Real PII fields are synthetic. Do not use for support training samples."),
    ("communications_preference", "Customer prefers email over phone. Phone number on file is for emergencies only."),
    ("vacation_hold", "Customer is on a 30-day vacation hold for SaaS organization {related_id} from 2026-05-01 to 2026-05-31."),
]


# Operational incident templates
_INCIDENT_TEMPLATES = [
    ("airline", "delayed_flight_data_update", "Delayed flight data update — JFK ATC issue", "Flight status updates for JFK departures were delayed by ~12 minutes between 09:14 and 10:42 UTC due to an ATC interface latency. Customer-facing status reflected stale data during that window; agents instructed to verify status manually."),
    ("airline", "weather_disruption", "Nor'easter — JFK widespread cancellations", "Severe weather disrupted JFK operations from 06:00 to 18:00 UTC. ~14,000 passengers affected. Refund/rebooking waiver granted automatically; manual handling for elite tier and multi-segment itineraries."),
    ("airline", "system_outage", "Booking system slow response", "Booking system saw elevated latency (p95 4.2s vs target 1s) for ~45 minutes due to upstream auth service degradation. Tickets not lost; users intermittently retried. Backend now healthy."),
    ("commerce", "payment_processor_outage", "Stripe outage — partial payment failures", "Stripe reported a regional incident from 14:00 to 14:45 UTC. ~3% of US-east orders failed at checkout. Affected customers received an apology email and a $10 voucher. No data loss."),
    ("commerce", "inventory_sync_delay", "Inventory sync lag — overselling risk", "Warehouse inventory sync to the product catalog lagged by ~6 hours due to a stuck Kafka consumer. Sold-out items briefly displayed as available. Pulled affected SKUs from catalog; ETL restarted."),
    ("commerce", "shipping_carrier_strike", "UPS strike — extended shipping windows", "UPS labor action affected ~22% of east-coast shipments in late April. Communicated revised delivery windows to all affected customers; offered free expedited reshipping if requested."),
    ("commerce", "tax_calculation_bug", "Tax calculation bug — overcharged orders", "Tax calculation rounded incorrectly for orders shipped to certain ZIP codes in CA. Affected ~2,300 orders; refunds processed automatically over 5 days."),
    ("saas", "billing_export_issue", "Billing CSV export missing line items", "Monthly billing CSV export omitted overage line items for ~8% of organizations in the Mar 2026 cycle. Re-issued corrected CSVs; emailed affected enterprise contacts."),
    ("saas", "api_rate_limit_misfire", "API rate limit misfire — false 429s", "Misconfiguration in the rate-limit service caused false 429 responses on ~0.2% of requests for 23 minutes. Auto-retry logic in the SDK masked the impact for most callers."),
    ("saas", "subscription_renewal_failure", "Renewal job failure — 41 orgs not renewed", "Scheduled renewal job failed silently on 2026-04-01 due to an SSL cert rotation issue. 41 organizations were technically lapsed for 6 hours. Re-renewed retroactively with no service interruption."),
    ("saas", "audit_log_gap", "Audit log gap — 18 minute window", "Audit log ingestion paused for 18 minutes due to a downstream Elastic indexing backlog. Backfilled from the WAL; integrity verified."),
    ("support", "support_sla_backlog", "Support SLA backlog after promo launch", "Support backlog exceeded 4-hour SLA for high-priority tickets during the spring promo launch. Brought 4 senior agents on overtime; backlog cleared within 36 hours."),
    ("support", "chat_widget_outage", "Chat widget outage — 35 minutes", "In-app chat widget was unreachable from 17:00 to 17:35 UTC due to a CDN misconfiguration. Tickets falling through to email channel; agent capacity adjusted."),
    ("crm", "marketing_email_misfire", "Marketing email misfire — opt-out leak", "Marketing email blast inadvertently included ~120 customers who had opted out of marketing. Sent apology + reaffirmed opt-out. DPO informed."),
    ("crm", "dsar_response_delay", "DSAR response delayed past 30-day SLA", "Two data subject access requests took 34 and 36 days respectively due to backlog. Both fulfilled; root cause: missing automation for SaaS-side data export."),
]


# Support resolution templates (the chatbot/agent reuses these)
_SUPPORT_RESOLUTION_TEMPLATES = [
    ("refund_delay", "Refund processing delay — standard response", "Hi {first_name}, thanks for your patience. Refunds typically take 5-7 business days to process, with an additional 1-2 billing cycles to appear on your statement. I checked your refund on {related_id} and confirmed it is in 'processing' status. Expected to land by {expected_date}.", False),
    ("refund_status_check", "Refund status — explicit check", "Hi {first_name}, your refund on {related_id} is currently {status}. Expected resolution date: {expected_date}. Refunds go to the original payment method automatically; no further action needed on your end.", False),
    ("baggage_lost_claim", "Lost baggage — claim filing", "Hi {first_name}, sorry to hear about the lost baggage on your flight. I've filed claim #{claim_id} with our baggage team. You should receive an update within 5 business days. Please keep your baggage tag if you still have it.", True),
    ("flight_change_self_service", "Self-service flight change instructions", "Hi {first_name}, you can change your flight self-service through the 'Manage Booking' section on our site. Use booking reference {pnr}. The change fee depends on cabin class and how close to departure the change is made.", False),
    ("flight_change_assisted", "Assisted flight change", "Hi {first_name}, I can help with your flight change on booking {pnr}. There is a change fee of ${fee} plus any fare difference. Would you like me to walk you through the available options?", False),
    ("baggage_allowance_question", "Baggage allowance question", "Hi {first_name}, on a {cabin_class} ticket for an {route_type} flight, your checked baggage allowance is {kg}kg. Cabin baggage: {cabin_kg}kg. Excess baggage is billed per kg at airport rates.", False),
    ("cancellation_request", "Cancellation request", "Hi {first_name}, I can cancel your booking {pnr}. Based on the fare class, you'll receive a refund of ${refund_amount} to your original payment method within 5-7 business days. Would you like me to proceed?", True),
    ("loyalty_points_missing", "Missing loyalty points claim", "Hi {first_name}, sorry for the inconvenience. I can help retroactively credit points for a recent flight. Could you confirm the booking reference and travel date? Claims must be filed within 6 months of the flight.", False),
    ("commerce_order_status", "Commerce order status update", "Hi {first_name}, your order {order_number} is currently {status}. Tracking number {tracking_number} via {carrier}. Estimated delivery: {estimated_delivery}.", False),
    ("commerce_order_not_received", "Order not received", "Hi {first_name}, I checked the tracking for {order_number} — it shows delivered on {delivery_date}. Please check with neighbours or your building's reception. If still missing, file a claim and we'll initiate an investigation with the carrier.", True),
    ("damaged_packaging", "Damaged packaging report", "Hi {first_name}, I'm sorry to hear about the damage. Please send photos within 48 hours of delivery to support@democorp.example. Once we receive them, we'll arrange a free replacement or full refund.", False),
    ("return_request", "Return request", "Hi {first_name}, return for order {order_number} approved. Use the prepaid label at the link above. Once we receive the item, your refund of ${refund_amount} will process within 3 business days.", False),
    ("saas_invoice_question", "Invoice question", "Hi {first_name}, invoice {invoice_number} for ${amount} was issued on {issued_at} and is due {due_at}. Line items breakdown: {line_summary}. Let me know which line you'd like to discuss.", False),
    ("saas_overage_explained", "Overage charges explained", "Hi {first_name}, the overage on org {org_id} for {month} is ${overage_amount}. This was calculated as {overage_calls} calls × ${rate}/1000 calls. The plan's grace period of 5% was already applied.", False),
    ("saas_subscription_cancel", "Subscription cancellation", "Hi {first_name}, I can cancel subscription on org {org_id}. Cancellation takes effect at the end of the current billing period ({end_date}). No refunds for the unused portion per the cancellation policy.", True),
    ("saas_seat_addition", "Adding seats", "Hi {first_name}, I can add {n} seats to org {org_id}'s plan. Pro-rated charge of ${prorated} will appear on your next invoice. Want me to proceed?", False),
    ("escalation_to_supervisor", "Escalation to supervisor", "Hi {first_name}, I'm escalating this to my supervisor for review. You'll receive a follow-up within {hours} hours. Thanks for your patience.", True),
    ("fraud_review_hold", "Fraud review hold", "Hi {first_name}, your account is under brief security review. We may ask for additional verification. We aim to clear reviews within 1 business day; thanks for understanding.", True),
    ("policy_link_only", "Policy reference reply", "Hi {first_name}, our {policy_type} policy is published at help.democorp.example/{policy_type}. The clauses most relevant to your question are {clause_keys}.", False),
    ("information_only_no_action", "Information-only response", "Hi {first_name}, this is informational and does not require action on your end. Let me know if you have follow-up questions.", False),
]


def _seed_knowledge(
    *,
    session: Session,
    counts: dict[str, int],
    customer_ids: list[int],
    rng: random.Random,
    fake: Faker,
    summary: dict[str, int],
) -> None:
    """Seed the textual knowledge tables (Phase 6B-2)."""

    # ---- policy_documents ----
    # Cycle through the curated 50-entry catalog and produce versioned copies
    # for the desired total. The catalog already has 50 distinct (domain,
    # policy_type) combos, so the small preset (50) is a 1:1 mapping.
    policy_target = counts["policy_documents"]
    policy_rows: list[dict[str, Any]] = []
    policy_clauses_planned: list[
        tuple[int, str, str, str, str, Optional[str], Optional[str]]
    ] = []  # (policy_id, clause_key, title, body, severity, applies_to, exceptions)

    today = datetime.now(timezone.utc).date()
    catalog_len = len(_POLICY_CATALOG)
    for i in range(policy_target):
        entry = _POLICY_CATALOG[i % catalog_len]
        version = 1 + (i // catalog_len)
        effective_from = today - timedelta(days=rng.randint(30, 720))
        is_active = (i // catalog_len == policy_target // catalog_len) or rng.random() < 0.85
        # 15% chance the document is superseded
        if not is_active:
            effective_to = effective_from + timedelta(days=rng.randint(60, 540))
        else:
            effective_to = None

        title_suffix = f" v{version}" if version > 1 else ""
        body_suffix = (
            ""
            if i < catalog_len
            else (
                "\n\n"
                + fake.paragraph(nb_sentences=rng.randint(3, 6))
                + "\n\nRevision history: minor wording updates and clarifications."
            )
        )
        policy_id = len(policy_rows) + 1
        policy_rows.append(
            {
                "id": policy_id,
                "domain": entry["domain"],
                "policy_type": entry["policy_type"],
                "title": f"{entry['title']}{title_suffix}",
                "version": version,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "is_active": is_active,
                "body": entry["body"] + body_suffix,
            }
        )
        # Stage clauses for this policy
        for clause in entry["clauses"]:
            policy_clauses_planned.append(
                (policy_id, *clause)
            )

    _bulk_insert(session, PolicyDocument, policy_rows)
    summary["policy_documents"] = len(policy_rows)

    # ---- policy_clauses ----
    clause_target = counts["policy_clauses"]
    clause_rows: list[dict[str, Any]] = []
    # Use the planned clauses first (these are domain-correct); if we need more,
    # cycle through them with minor wording variation.
    planned = list(policy_clauses_planned)
    rng.shuffle(planned)
    while len(clause_rows) < clause_target:
        for entry in planned:
            if len(clause_rows) >= clause_target:
                break
            policy_id, clause_key, title, body, severity, applies_to, exceptions = entry
            # Add a faint paraphrasing tail beyond first cycle so the text isn't
            # byte-identical (helps full-text search realism).
            cycle = len(clause_rows) // len(planned)
            body_out = body
            if cycle >= 1:
                body_out = (
                    body
                    + " "
                    + fake.sentence(nb_words=rng.randint(8, 18))
                )
            clause_rows.append(
                {
                    "id": len(clause_rows) + 1,
                    "policy_document_id": policy_id,
                    "clause_key": clause_key if cycle == 0 else f"{clause_key}_v{cycle + 1}",
                    "title": title,
                    "body": body_out,
                    "severity": severity,
                    "applies_to": applies_to,
                    "exceptions": exceptions,
                }
            )
    _bulk_insert(session, PolicyClause, clause_rows)
    summary["policy_clauses"] = len(clause_rows)

    # ---- product_warranty_terms ----
    warranty_target = counts["product_warranty_terms"]
    product_count = counts["products"]
    # Note: products are seeded with explicit ids 1..product_count
    warranty_rows: list[dict[str, Any]] = []
    for i in range(warranty_target):
        product_id = (i % product_count) + 1
        tpl = _WARRANTY_TEMPLATES[i % len(_WARRANTY_TEMPLATES)]
        # Vary duration slightly across rows (medium/large) to add realism
        duration = tpl["duration_months"]
        if i >= len(_WARRANTY_TEMPLATES):
            # Small jitter beyond first cycle
            duration = max(0, duration + rng.choice([-3, 0, 0, 0, 3, 6]))
        # Add a one-sentence tail for variety beyond the first product
        extra = (
            ""
            if i < product_count
            else " " + fake.sentence(nb_words=rng.randint(10, 18))
        )
        warranty_rows.append(
            {
                "id": i + 1,
                "product_id": product_id,
                "warranty_type": tpl["warranty_type"],
                "duration_months": duration,
                "body": tpl["body_template"] + extra,
                "exclusions": tpl["exclusions"],
            }
        )
    _bulk_insert(session, ProductWarrantyTerms, warranty_rows)
    summary["product_warranty_terms"] = len(warranty_rows)

    # ---- product_return_rules ----
    rule_target = counts["product_return_rules"]
    category_count = counts["product_categories"]
    rule_rows: list[dict[str, Any]] = []
    for i in range(rule_target):
        category_id = (i % category_count) + 1
        tpl = _RETURN_RULE_TEMPLATES[i % len(_RETURN_RULE_TEMPLATES)]
        extra = (
            ""
            if i < len(_RETURN_RULE_TEMPLATES)
            else " " + fake.sentence(nb_words=rng.randint(10, 18))
        )
        rule_rows.append(
            {
                "id": i + 1,
                "product_category_id": category_id,
                "rule_name": tpl["rule_name"],
                "body": tpl["body"] + extra,
                "opened_item_allowed": tpl["opened_item_allowed"],
                "return_window_days": tpl["return_window_days"],
                "restocking_fee_percent": tpl["restocking_fee_percent"],
                "exceptions": tpl["exceptions"],
            }
        )
    _bulk_insert(session, ProductReturnRule, rule_rows)
    summary["product_return_rules"] = len(rule_rows)

    # ---- internal_agent_notes ----
    # Distribute notes across customers; a customer may have multiple notes.
    note_target = counts["internal_agent_notes"]
    # Realistic related-entity pools (using actual seeded id ranges).
    related_pools = [
        ("booking", counts["bookings"]),
        ("order", counts["commerce_orders"]),
        ("ticket", counts["support_tickets"]),
        ("invoice", counts["invoices"]),
        ("subscription", counts["subscriptions"]),
        (None, 0),  # standalone note
    ]
    note_rows: list[dict[str, Any]] = []
    for i in range(note_target):
        cust_id = customer_ids[rng.randrange(len(customer_ids))]
        note_type, tpl_body = _NOTE_TEMPLATES[i % len(_NOTE_TEMPLATES)]
        related_type, max_id = rng.choice(related_pools)
        related_id: Optional[int] = (
            rng.randint(1, max_id) if related_type is not None and max_id > 0 else None
        )
        body = tpl_body.format(
            related_type=related_type or "n/a",
            related_id=related_id if related_id is not None else "n/a",
        )
        # Add a contextual tail beyond the first pass for variety.
        if i >= len(_NOTE_TEMPLATES):
            body = body + " " + fake.sentence(nb_words=rng.randint(12, 24))
        note_rows.append(
            {
                "id": i + 1,
                "customer_id": cust_id,
                "related_type": related_type,
                "related_id": related_id,
                "note_type": note_type,
                "body": body,
            }
        )
    _bulk_insert(session, InternalAgentNote, note_rows)
    summary["internal_agent_notes"] = len(note_rows)

    # ---- operational_incidents ----
    incident_target = counts["operational_incidents"]
    incident_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for i in range(incident_target):
        domain, kind, title, body = _INCIDENT_TEMPLATES[i % len(_INCIDENT_TEMPLATES)]
        started = now - timedelta(days=rng.randint(0, 365), hours=rng.randint(0, 23))
        resolved: Optional[datetime] = None
        if rng.random() < 0.92:
            resolved = started + timedelta(
                minutes=rng.randint(15, 60 * 18)
            )
        affected = {
            "affected_count": rng.randint(50, 50_000),
            "internal_severity": rng.choice(["sev1", "sev2", "sev3"]),
            "comms_sent": rng.random() < 0.7,
        }
        # Body variety beyond first cycle
        body_out = body
        if i >= len(_INCIDENT_TEMPLATES):
            body_out = (
                body
                + " "
                + fake.sentence(nb_words=rng.randint(12, 22))
            )
        incident_rows.append(
            {
                "id": i + 1,
                "domain": domain,
                "incident_type": kind,
                "title": title,
                "body": body_out,
                "started_at": started,
                "resolved_at": resolved,
                "affected_entities_json": affected,
            }
        )
    _bulk_insert(session, OperationalIncident, incident_rows)
    summary["operational_incidents"] = len(incident_rows)

    # ---- support_resolution_templates ----
    template_target = counts["support_resolution_templates"]
    template_rows: list[dict[str, Any]] = []
    for i in range(template_target):
        category, title, body, esc = _SUPPORT_RESOLUTION_TEMPLATES[
            i % len(_SUPPORT_RESOLUTION_TEMPLATES)
        ]
        # Distinct titles when cycling
        title_suffix = "" if i < len(_SUPPORT_RESOLUTION_TEMPLATES) else f" — variant {i // len(_SUPPORT_RESOLUTION_TEMPLATES) + 1}"
        template_rows.append(
            {
                "id": i + 1,
                "category": category,
                "title": title + title_suffix,
                "body": body,
                "escalation_required": esc,
            }
        )
    _bulk_insert(session, SupportResolutionTemplate, template_rows)
    summary["support_resolution_templates"] = len(template_rows)
