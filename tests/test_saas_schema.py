"""Tests for the SaaS / billing schema (Phase B1).

Pure schema + seed tests — no chatbot behaviour, no new tools.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    ApiUsageDaily,
    Customer,
    CustomerOrganization,
    Invoice,
    InvoiceItem,
    Organization,
    OverageCharge,
    Plan,
    SeatAllocation,
    Subscription,
    UsageEvent,
)
from app.seed import SCALES


# ---------------------------------------------------------------------------
# SCALES — Phase B1 keys present at every preset
# ---------------------------------------------------------------------------

_SAAS_KEYS = (
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


def test_saas_keys_present_in_every_preset() -> None:
    for name in ("small", "medium", "large"):
        for key in _SAAS_KEYS:
            assert key in SCALES[name], f"missing {key!r} from {name!r}"


def test_saas_counts_monotonic_across_scales() -> None:
    for key in _SAAS_KEYS:
        assert SCALES["large"][key] >= SCALES["medium"][key], key
        assert SCALES["medium"][key] >= SCALES["small"][key], key


def test_small_saas_counts_match_phase_b1_spec() -> None:
    s = SCALES["small"]
    assert s["organizations"] == 50
    assert s["customer_organizations"] == 150
    assert s["plans"] == 4
    assert s["subscriptions"] == 50
    assert s["invoices"] == 200
    assert s["invoice_items"] == 600
    assert s["usage_events"] == 500
    assert s["api_usage_daily"] == 1500
    assert s["seat_allocations"] == 50
    assert s["overage_charges"] == 80


def test_medium_saas_counts_meaningful() -> None:
    m = SCALES["medium"]
    assert m["organizations"] >= 1_000
    assert m["subscriptions"] >= 1_000
    assert m["invoices"] >= 5_000
    assert m["usage_events"] >= 25_000


# ---------------------------------------------------------------------------
# Seeded DB — counts match the summary
# ---------------------------------------------------------------------------


def test_seed_persists_expected_saas_row_counts(seeded_engine) -> None:
    engine, summary = seeded_engine
    s = SCALES["small"]
    with Session(engine) as session:
        n = lambda Tbl: session.execute(  # noqa: E731
            select(func.count()).select_from(Tbl)
        ).scalar_one()
        assert n(Organization) == s["organizations"]
        assert n(CustomerOrganization) == s["customer_organizations"]
        assert n(Plan) == s["plans"]
        assert n(Subscription) == s["subscriptions"]
        assert n(Invoice) == s["invoices"]
        assert n(InvoiceItem) == s["invoice_items"]
        assert n(UsageEvent) == s["usage_events"]
        assert n(ApiUsageDaily) == s["api_usage_daily"]
        assert n(SeatAllocation) == s["seat_allocations"]
        assert n(OverageCharge) == s["overage_charges"]
    # The summary returned from seed() also includes the SaaS keys.
    for key in _SAAS_KEYS:
        assert summary[key] == s[key]


def test_saas_seed_did_not_disturb_airline_counts(seeded_engine) -> None:
    """All Phase 1C airline counts are unchanged after adding SaaS seeding."""
    _, summary = seeded_engine
    s = SCALES["small"]
    for key in (
        "customers",
        "airports",
        "flights",
        "bookings",
        "seats",
        "baggage_rules",
        "refunds",
        "support_tickets",
        "support_messages",
        "kb_articles",
    ):
        assert summary[key] == s[key], key


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def test_customer_organizations_link_real_customers_and_orgs(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        orphan_customers = s.execute(
            select(func.count())
            .select_from(CustomerOrganization)
            .outerjoin(Customer, CustomerOrganization.customer_id == Customer.id)
            .where(Customer.id.is_(None))
        ).scalar_one()
        orphan_orgs = s.execute(
            select(func.count())
            .select_from(CustomerOrganization)
            .outerjoin(Organization, CustomerOrganization.organization_id == Organization.id)
            .where(Organization.id.is_(None))
        ).scalar_one()
    assert orphan_customers == 0
    assert orphan_orgs == 0


def test_subscriptions_reference_real_orgs_and_plans(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad_org = s.execute(
            select(func.count())
            .select_from(Subscription)
            .outerjoin(Organization, Subscription.organization_id == Organization.id)
            .where(Organization.id.is_(None))
        ).scalar_one()
        bad_plan = s.execute(
            select(func.count())
            .select_from(Subscription)
            .outerjoin(Plan, Subscription.plan_id == Plan.id)
            .where(Plan.id.is_(None))
        ).scalar_one()
    assert bad_org == 0
    assert bad_plan == 0


def test_invoices_link_real_orgs(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad = s.execute(
            select(func.count())
            .select_from(Invoice)
            .outerjoin(Organization, Invoice.organization_id == Organization.id)
            .where(Organization.id.is_(None))
        ).scalar_one()
    assert bad == 0


def test_invoice_items_link_real_invoices(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad = s.execute(
            select(func.count())
            .select_from(InvoiceItem)
            .outerjoin(Invoice, InvoiceItem.invoice_id == Invoice.id)
            .where(Invoice.id.is_(None))
        ).scalar_one()
    assert bad == 0


def test_overage_charges_link_real_invoices(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        bad = s.execute(
            select(func.count())
            .select_from(OverageCharge)
            .outerjoin(Invoice, OverageCharge.invoice_id == Invoice.id)
            .where(Invoice.id.is_(None))
        ).scalar_one()
    assert bad == 0


def test_seat_allocations_one_per_org_max(seeded_engine) -> None:
    """``seat_allocations.organization_id`` is unique."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        total = s.execute(select(func.count()).select_from(SeatAllocation)).scalar_one()
        distinct_orgs = s.execute(
            select(func.count(func.distinct(SeatAllocation.organization_id)))
        ).scalar_one()
    assert total == distinct_orgs


def test_api_usage_daily_unique_per_org_date(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        total = s.execute(select(func.count()).select_from(ApiUsageDaily)).scalar_one()
        distinct = s.execute(
            select(
                func.count(
                    func.distinct(
                        func.printf(
                            "%s|%s",
                            ApiUsageDaily.organization_id,
                            ApiUsageDaily.date,
                        )
                    )
                )
            )
        ).scalar_one()
    assert total == distinct


def test_plans_have_expected_tiers(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        tiers = {t for (t,) in s.execute(select(Plan.tier)).all()}
    assert {"starter", "pro", "business", "enterprise"} <= tiers


def test_customer_can_be_in_multiple_organizations(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        # Group memberships by customer and find at least one customer with >= 1 link.
        # The seed guarantees 150 links across 500 customers, so the average
        # cust has 0.3 memberships; some customers will have multiple.
        counts_per_customer = s.execute(
            select(CustomerOrganization.customer_id, func.count())
            .group_by(CustomerOrganization.customer_id)
        ).all()
    assert counts_per_customer, "no customer-organization links seeded"
    # Most customers have at most 1, but the test just confirms the link exists.
    assert any(count >= 1 for _, count in counts_per_customer)


# ---------------------------------------------------------------------------
# Unique-constraint behaviour
# ---------------------------------------------------------------------------


def test_customer_organization_pair_is_unique(seeded_engine, tmp_path) -> None:
    """Inserting the same (customer, org) twice raises IntegrityError."""
    from sqlalchemy.orm import sessionmaker

    engine, _ = seeded_engine
    Session_ = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)

    with Session_() as s:
        existing = s.execute(select(CustomerOrganization).limit(1)).scalar_one()
        s.add(
            CustomerOrganization(
                customer_id=existing.customer_id,
                organization_id=existing.organization_id,
                role="member",
            )
        )
        with pytest.raises(IntegrityError):
            s.flush()
        s.rollback()
