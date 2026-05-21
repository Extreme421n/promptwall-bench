"""Tests for the 7 textual-retrieval tools added in Phase 6B-4."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Customer,
    OperationalIncident,
    PolicyDocument,
    Product,
    ProductCategory,
)
from app.tools import (
    ResourceNotFoundError,
    ToolValidationError,
    get_active_policy,
    get_policy_clause,
    get_product_warranty_terms,
    get_support_resolution_template,
    list_policy_versions,
    search_internal_agent_notes,
    search_operational_incidents,
    search_policy_documents,
    search_return_rules,
)


# ---------------------------------------------------------------------------
# search_policy_documents (rewritten in 6B-4)
# ---------------------------------------------------------------------------


def test_search_policy_documents_returns_spec_fields(seeded_session: Session) -> None:
    out = search_policy_documents.call(seeded_session, {"query": "refund"})
    assert out["count"] >= 1
    for d in out["documents"]:
        for k in ("id", "title", "domain", "policy_type", "version", "excerpt", "effective_from"):
            assert k in d, f"missing field: {k!r}"


def test_search_policy_documents_filters_domain(seeded_session: Session) -> None:
    out = search_policy_documents.call(
        seeded_session, {"query": "policy", "domain": "airline"}
    )
    assert out["count"] >= 1
    assert all(d["domain"] == "airline" for d in out["documents"])


def test_search_policy_documents_filters_policy_type(seeded_session: Session) -> None:
    out = search_policy_documents.call(
        seeded_session, {"query": "policy", "policy_type": "baggage_policy"}
    )
    assert all(d["policy_type"] == "baggage_policy" for d in out["documents"])


# ---------------------------------------------------------------------------
# get_policy_clause (rewritten in 6B-4)
# ---------------------------------------------------------------------------


def test_get_policy_clause_by_query(seeded_session: Session) -> None:
    out = get_policy_clause.call(seeded_session, {"query": "refund"})
    assert out["count"] >= 1
    for c in out["clauses"]:
        for k in (
            "clause_id", "policy_document_id", "policy_title", "policy_domain",
            "policy_type", "clause_key", "title", "body", "severity",
            "applies_to", "exceptions",
        ):
            assert k in c, f"missing clause field: {k!r}"


def test_get_policy_clause_back_compat_policy_topic(seeded_session: Session) -> None:
    out = get_policy_clause.call(seeded_session, {"policy_topic": "refund"})
    assert out["count"] >= 1


def test_get_policy_clause_by_clause_key(seeded_session: Session) -> None:
    out = get_policy_clause.call(seeded_session, {"clause_key": "eligibility"})
    assert out["count"] >= 1
    assert all(c["clause_key"].startswith("eligibility") for c in out["clauses"])


def test_get_policy_clause_no_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_policy_clause.call(seeded_session, {})


# ---------------------------------------------------------------------------
# search_return_rules
# ---------------------------------------------------------------------------


def test_search_return_rules_returns_matches(seeded_session: Session) -> None:
    out = search_return_rules.call(seeded_session, {"query": "return"})
    assert out["count"] >= 1
    for r in out["rules"]:
        assert "rule_name" in r
        assert "opened_item_allowed" in r
        assert "return_window_days" in r
        assert "restocking_fee_percent" in r
        assert isinstance(r["product_category_name"], str)


def test_search_return_rules_category_filter(seeded_session: Session) -> None:
    out = search_return_rules.call(
        seeded_session,
        {"query": "return", "product_category": "Electronics"},
    )
    # Some categories exist; filter is partial-match on name.
    for r in out["rules"]:
        assert "Electronics".lower() in r["product_category_name"].lower()


def test_search_return_rules_missing_query(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_return_rules.call(seeded_session, {})


def test_search_return_rules_query_too_short(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_return_rules.call(seeded_session, {"query": "a"})


# ---------------------------------------------------------------------------
# get_product_warranty_terms
# ---------------------------------------------------------------------------


def test_get_warranty_by_sku(seeded_session: Session) -> None:
    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    out = get_product_warranty_terms.call(seeded_session, {"sku": sku})
    assert out["count"] >= 1
    for t in out["terms"]:
        assert t["product_sku"] == sku
        assert "warranty_type" in t
        assert t["duration_months"] >= 0


def test_get_warranty_by_product_id(seeded_session: Session) -> None:
    pid = seeded_session.execute(select(Product.id).limit(1)).scalar_one()
    out = get_product_warranty_terms.call(seeded_session, {"product_id": pid})
    assert out["count"] >= 1


def test_get_warranty_unknown_product(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_product_warranty_terms.call(seeded_session, {"sku": "SKU-NO-SUCH-99"})


def test_get_warranty_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_product_warranty_terms.call(seeded_session, {})


# ---------------------------------------------------------------------------
# search_internal_agent_notes
# ---------------------------------------------------------------------------


def test_search_notes_for_customer(seeded_session: Session) -> None:
    cust = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    out = search_internal_agent_notes.call(seeded_session, {"customer_id": cust})
    # Customer may have 0+ notes; just verify shape.
    assert "notes" in out
    for n in out["notes"]:
        assert n["customer_id"] == cust
        assert "note_type" in n
        assert "body_excerpt" in n


def test_search_notes_with_query(seeded_session: Session) -> None:
    cust = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    out = search_internal_agent_notes.call(
        seeded_session, {"customer_id": cust, "query": "customer"}
    )
    assert "notes" in out


def test_search_notes_unknown_customer(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        search_internal_agent_notes.call(
            seeded_session, {"customer_id": 99_999_999}
        )


def test_search_notes_missing_customer_id(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_internal_agent_notes.call(seeded_session, {})


# ---------------------------------------------------------------------------
# search_operational_incidents
# ---------------------------------------------------------------------------


def test_search_incidents_by_query(seeded_session: Session) -> None:
    out = search_operational_incidents.call(seeded_session, {"query": "delay"})
    assert "incidents" in out


def test_search_incidents_by_domain(seeded_session: Session) -> None:
    out = search_operational_incidents.call(
        seeded_session, {"domain": "airline"}
    )
    assert out["count"] >= 1
    assert all(i["domain"] == "airline" for i in out["incidents"])


def test_search_incidents_active_only(seeded_session: Session) -> None:
    out = search_operational_incidents.call(
        seeded_session, {"domain": "airline", "active_only": True}
    )
    assert all(i["resolved_at"] is None for i in out["incidents"])


def test_search_incidents_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        search_operational_incidents.call(seeded_session, {})


# ---------------------------------------------------------------------------
# get_support_resolution_template
# ---------------------------------------------------------------------------


def test_get_template_by_category(seeded_session: Session) -> None:
    out = get_support_resolution_template.call(
        seeded_session, {"category": "refund_delay"}
    )
    assert out["count"] >= 1
    for t in out["templates"]:
        assert t["category"] == "refund_delay"
        assert "escalation_required" in t


def test_get_template_by_query(seeded_session: Session) -> None:
    out = get_support_resolution_template.call(
        seeded_session, {"query": "refund"}
    )
    assert out["count"] >= 1


def test_get_template_missing_input(seeded_session: Session) -> None:
    with pytest.raises(ToolValidationError):
        get_support_resolution_template.call(seeded_session, {})


# ---------------------------------------------------------------------------
# list_policy_versions (bonus)
# ---------------------------------------------------------------------------


def test_list_policy_versions_returns_at_least_one(seeded_session: Session) -> None:
    out = list_policy_versions.call(
        seeded_session, {"domain": "airline", "policy_type": "refund_policy"}
    )
    assert out["count"] >= 1
    versions = [v["version"] for v in out["versions"]]
    assert versions == sorted(versions, reverse=True), "versions must be DESC"


def test_list_policy_versions_unknown(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        list_policy_versions.call(
            seeded_session,
            {"domain": "airline", "policy_type": "no_such_policy"},
        )


# ---------------------------------------------------------------------------
# get_active_policy (bonus)
# ---------------------------------------------------------------------------


def test_get_active_policy_returns_active_version(seeded_session: Session) -> None:
    out = get_active_policy.call(
        seeded_session,
        {"domain": "airline", "policy_type": "refund_policy"},
    )
    assert out["is_active"] is True
    assert out["domain"] == "airline"
    assert out["policy_type"] == "refund_policy"
    assert len(out["body"]) >= 80


def test_get_active_policy_unknown(seeded_session: Session) -> None:
    with pytest.raises(ResourceNotFoundError):
        get_active_policy.call(
            seeded_session,
            {"domain": "airline", "policy_type": "no_such_policy"},
        )


# ---------------------------------------------------------------------------
# tool_invocations integration: every new tool logs via ToolExecutor
# ---------------------------------------------------------------------------


def test_new_tools_log_through_tool_executor(seeded_session: Session) -> None:
    """Each of the 7 new tools is reachable via ToolExecutor with logging."""
    from sqlalchemy import func
    from app.models import ToolInvocation
    from app.services import ToolExecutor

    executor = ToolExecutor(seeded_session)
    before = seeded_session.execute(
        select(func.count()).select_from(ToolInvocation)
    ).scalar_one()

    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()

    invocations = [
        ("search_return_rules", {"query": "return"}),
        ("get_product_warranty_terms", {"sku": sku}),
        ("search_internal_agent_notes", {"customer_id": cust_id}),
        ("search_operational_incidents", {"domain": "airline"}),
        ("get_support_resolution_template", {"category": "refund_delay"}),
        ("list_policy_versions", {"domain": "airline", "policy_type": "refund_policy"}),
        ("get_active_policy", {"domain": "airline", "policy_type": "refund_policy"}),
    ]
    for name, payload in invocations:
        result = executor.execute_tool(trace_id=None, tool_name=name, input_json=payload)
        assert result.success is True, f"{name} failed: {result.error_message}"
        assert result.evidence_id is not None
        assert result.evidence_id.startswith("ev_")

    after = seeded_session.execute(
        select(func.count()).select_from(ToolInvocation)
    ).scalar_one()
    # 7 tool calls → 7 new tool_invocation rows.
    assert after - before == 7
