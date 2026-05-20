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
    Invoice,
    InvoiceItem,
    KBArticle,
    LoyaltyAccount,
    Organization,
    OverageCharge,
    Plan,
    Product,
    ProductAttribute,
    ProductCategory,
    ProductInventory,
    ProductPrice,
    Refund,
    Seat,
    SeatAllocation,
    Shipment,
    Subscription,
    SupportMessage,
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
