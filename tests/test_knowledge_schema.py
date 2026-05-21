"""Tests for the textual knowledge schema (Phase 6B-1)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Airport,
    Customer,
    InternalAgentNote,
    OperationalIncident,
    PolicyClause,
    PolicyDocument,
    Product,
    ProductCategory,
    ProductReturnRule,
    ProductWarrantyTerms,
    SupportResolutionTemplate,
)


# ---------------------------------------------------------------------------
# Schema-level invariants
# ---------------------------------------------------------------------------


def test_seven_new_tables_present_in_metadata() -> None:
    from app import models

    expected = {
        "policy_documents",
        "policy_clauses",
        "product_warranty_terms",
        "product_return_rules",
        "internal_agent_notes",
        "operational_incidents",
        "support_resolution_templates",
    }
    present = {t.name for t in models.Base.metadata.tables.values()}
    missing = expected - present
    assert not missing, f"missing tables: {missing}"


def test_policy_documents_has_composite_index(db: Session) -> None:
    """Index ix_policy_documents_domain_type_active is declared on the model."""
    tbl = PolicyDocument.__table__
    index_names = {idx.name for idx in tbl.indexes}
    assert "ix_policy_documents_domain_type_active" in index_names


def test_operational_incidents_has_composite_index() -> None:
    tbl = OperationalIncident.__table__
    index_names = {idx.name for idx in tbl.indexes}
    assert "ix_operational_incidents_domain_type" in index_names


# ---------------------------------------------------------------------------
# Insert + query round trips for each table
# ---------------------------------------------------------------------------


def test_policy_document_insert_and_query(db: Session) -> None:
    doc = PolicyDocument(
        domain="airline",
        title="Refund Policy v1",
        policy_type="refund_policy",
        version=1,
        effective_from=date(2026, 1, 1),
        is_active=True,
        body="Refunds are eligible if cancelled at least 24 hours before departure...",
    )
    db.add(doc)
    db.flush()

    fetched = db.execute(
        select(PolicyDocument).where(PolicyDocument.title == "Refund Policy v1")
    ).scalar_one()
    assert fetched.domain == "airline"
    assert fetched.policy_type == "refund_policy"
    assert fetched.is_active is True
    assert fetched.effective_to is None


def test_policy_document_with_clauses_cascade(db: Session) -> None:
    doc = PolicyDocument(
        domain="commerce",
        title="Return Policy 2026",
        policy_type="return_policy",
        version=2,
        effective_from=date(2026, 2, 1),
        body="Customers may return products within 30 days...",
    )
    doc.clauses = [
        PolicyClause(
            clause_key="opened_items",
            title="Opened items",
            body="Opened consumables are non-returnable.",
            severity="high",
            applies_to="consumables",
        ),
        PolicyClause(
            clause_key="restocking_fee",
            title="Restocking fee",
            body="A 15% restocking fee applies to returns of opened electronics.",
            severity="normal",
            applies_to="electronics",
        ),
    ]
    db.add(doc)
    db.flush()

    fetched = db.execute(
        select(PolicyDocument).where(PolicyDocument.title == "Return Policy 2026")
    ).scalar_one()
    assert len(fetched.clauses) == 2
    keys = {c.clause_key for c in fetched.clauses}
    assert keys == {"opened_items", "restocking_fee"}

    # Cascade delete: removing the document removes its clauses.
    doc_id = fetched.id
    db.delete(fetched)
    db.flush()

    remaining = db.execute(
        select(func.count())
        .select_from(PolicyClause)
        .where(PolicyClause.policy_document_id == doc_id)
    ).scalar_one()
    assert remaining == 0


def test_product_warranty_terms_insert_and_query(db: Session) -> None:
    # Needs a Product and a ProductCategory parent.
    cat = ProductCategory(name="Electronics")
    db.add(cat)
    db.flush()
    product = Product(
        sku="SKU-TEST-WARR-001",
        name="Test Headphones",
        category_id=cat.id,
        description="Test product for warranty terms",
        brand="Acme",
        is_active=True,
    )
    db.add(product)
    db.flush()

    warranty = ProductWarrantyTerms(
        product_id=product.id,
        warranty_type="manufacturer",
        duration_months=24,
        body=(
            "Manufacturer warranty covering defects in materials and workmanship "
            "for 24 months from date of purchase."
        ),
        exclusions="Physical damage, water exposure, modifications.",
    )
    db.add(warranty)
    db.flush()

    fetched = db.execute(
        select(ProductWarrantyTerms).where(ProductWarrantyTerms.product_id == product.id)
    ).scalar_one()
    assert fetched.warranty_type == "manufacturer"
    assert fetched.duration_months == 24
    assert "warranty" in fetched.body.lower()


def test_product_return_rule_insert_and_query(db: Session) -> None:
    cat = ProductCategory(name="Apparel")
    db.add(cat)
    db.flush()

    rule = ProductReturnRule(
        product_category_id=cat.id,
        rule_name="Standard apparel return",
        body="Unworn apparel may be returned within 30 days with original tags.",
        opened_item_allowed=False,
        return_window_days=30,
        restocking_fee_percent=Decimal("0.00"),
        exceptions="Final sale items, swimwear, and undergarments.",
    )
    db.add(rule)
    db.flush()

    fetched = db.execute(
        select(ProductReturnRule).where(ProductReturnRule.product_category_id == cat.id)
    ).scalar_one()
    assert fetched.return_window_days == 30
    assert fetched.opened_item_allowed is False
    assert fetched.restocking_fee_percent == Decimal("0.00")
    assert "swimwear" in (fetched.exceptions or "")


def test_internal_agent_note_insert_and_query(db: Session) -> None:
    cust = Customer(
        external_customer_id="CUST-TEST-NOTE-1",
        full_name="Test Customer",
        email="testnote@example.com",
    )
    db.add(cust)
    db.flush()

    note = InternalAgentNote(
        customer_id=cust.id,
        related_type="booking",
        related_id=42,
        note_type="vip_handling",
        body=(
            "Customer escalated due to repeated flight delays on JFK-LHR route. "
            "Approved one-time courtesy upgrade. Do not repeat without supervisor sign-off."
        ),
    )
    db.add(note)
    db.flush()

    fetched = db.execute(
        select(InternalAgentNote).where(InternalAgentNote.customer_id == cust.id)
    ).scalar_one()
    assert fetched.related_type == "booking"
    assert fetched.related_id == 42
    assert fetched.note_type == "vip_handling"
    assert "courtesy upgrade" in fetched.body


def test_internal_agent_note_polymorphic_pointer_nullable(db: Session) -> None:
    """An agent note can stand alone, with no related entity."""
    cust = Customer(
        external_customer_id="CUST-TEST-NOTE-2",
        full_name="Standalone Note Customer",
        email="standalone@example.com",
    )
    db.add(cust)
    db.flush()

    note = InternalAgentNote(
        customer_id=cust.id,
        note_type="general",
        body="Prefers email over phone outreach.",
    )
    db.add(note)
    db.flush()

    fetched = db.execute(
        select(InternalAgentNote).where(InternalAgentNote.id == note.id)
    ).scalar_one()
    assert fetched.related_type is None
    assert fetched.related_id is None


def test_operational_incident_insert_and_query(db: Session) -> None:
    started = datetime(2026, 5, 15, 9, 30, tzinfo=timezone.utc)
    incident = OperationalIncident(
        domain="airline",
        incident_type="weather_disruption",
        title="JFK widespread cancellations — Nor'easter",
        body=(
            "Severe weather grounded outbound flights from JFK between 09:00 and 18:00 UTC. "
            "Affected ~14,000 passengers. Refund/rebooking waivers granted automatically."
        ),
        started_at=started,
        resolved_at=started + timedelta(hours=12),
        affected_entities_json={
            "airports": ["JFK"],
            "affected_flights": 312,
            "approx_passengers": 14_000,
        },
    )
    db.add(incident)
    db.flush()

    fetched = db.execute(
        select(OperationalIncident).where(
            OperationalIncident.title.like("JFK widespread%")
        )
    ).scalar_one()
    assert fetched.domain == "airline"
    assert fetched.affected_entities_json["affected_flights"] == 312
    assert fetched.resolved_at is not None


def test_support_resolution_template_insert_and_query(db: Session) -> None:
    tpl = SupportResolutionTemplate(
        category="refund_delay",
        title="Refund processing delay — standard response",
        body=(
            "Hi {first_name}, thanks for your patience. Refunds typically take "
            "5-7 business days to process, and an additional 1-2 billing cycles "
            "to appear on your statement..."
        ),
        escalation_required=False,
    )
    db.add(tpl)
    db.flush()

    fetched = db.execute(
        select(SupportResolutionTemplate).where(
            SupportResolutionTemplate.category == "refund_delay"
        )
    ).scalar_one()
    assert fetched.escalation_required is False
    assert "{first_name}" in fetched.body


# ---------------------------------------------------------------------------
# Relationship traversal
# ---------------------------------------------------------------------------


def test_policy_clauses_back_reference_to_document(db: Session) -> None:
    doc = PolicyDocument(
        domain="saas",
        title="Overage Policy",
        policy_type="overage_policy",
        version=3,
        effective_from=date(2026, 3, 1),
        body="API call overages are charged at the plan's per-1000-call rate...",
    )
    clause = PolicyClause(
        clause_key="grace_period",
        title="Grace period",
        body="A 5% over-quota grace period applies before charges begin.",
    )
    doc.clauses = [clause]
    db.add(doc)
    db.flush()

    fetched_clause = db.execute(
        select(PolicyClause).where(PolicyClause.clause_key == "grace_period")
    ).scalar_one()
    assert fetched_clause.policy_document is not None
    assert fetched_clause.policy_document.title == "Overage Policy"
    assert fetched_clause.policy_document.policy_type == "overage_policy"
