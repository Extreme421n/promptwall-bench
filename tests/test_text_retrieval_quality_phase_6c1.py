"""Phase 6C-1 — text retrieval quality tests.

These tests pin down that realistic user wording (with stopwords, plurals, and
common synonyms) actually finds the relevant policy / knowledge text in the
seeded DB. The tools must:

  * normalize the query (lowercase, strip punctuation),
  * synonym-expand (token + phrase),
  * search across multiple fields,
  * attach match_score / match_reason / matched_fields / excerpt to every row,
  * fall back to a policy-type-inferred lookup when free-text returns nothing.

The expansion module ``app.tools._text_search`` is unit-tested in isolation
first; then the eight spec phrases are exercised end-to-end through each tool.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.tools import (
    get_policy_clause,
    get_product_warranty_terms,
    get_support_resolution_template,
    search_internal_agent_notes,
    search_operational_incidents,
    search_policy_documents,
    search_return_rules,
)
from app.tools._text_search import (
    _singularize,
    expand_query,
    infer_policy_types,
    make_excerpt,
    normalize,
    score_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ranked_first(items: list) -> dict:
    """Return the first item dict from a tool-output list."""
    assert items, "tool returned zero rows"
    first = items[0]
    return first.model_dump() if hasattr(first, "model_dump") else first


def _assert_match_fields(item: dict) -> None:
    """Every Phase 6C-1 tool item must expose these four fields."""
    for k in ("match_score", "match_reason", "matched_fields", "excerpt"):
        assert k in item, f"missing field {k!r}: {sorted(item)}"
    assert 0.0 <= item["match_score"] <= 1.0, item["match_score"]
    assert isinstance(item["matched_fields"], list)
    assert isinstance(item["excerpt"], str) and item["excerpt"]


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------


def test_normalize_lowercases_strips_punctuation() -> None:
    assert normalize("Can I RETURN an opened electronic product?") == (
        "can i return an opened electronic product"
    )
    assert normalize("  Multiple   spaces, and? punct!! ") == (
        "multiple spaces and punct"
    )
    # Hyphens preserved (carry-on, irrops etc.).
    assert "carry-on" in normalize("Carry-On baggage allowance")


def test_normalize_handles_empty() -> None:
    assert normalize("") == ""
    assert normalize(None) == ""


def test_singularize_safe() -> None:
    assert _singularize("policies") == "policy"
    assert _singularize("electronics") == "electronic"
    assert _singularize("returns") == "returns"  # protected — lemma owns the s
    assert _singularize("dispute") == "dispute"  # already singular


def test_expand_query_includes_full_phrase_and_tokens() -> None:
    terms = expand_query("Can I return an opened electronic product?")
    # Full normalized query is term #0 — highest specificity.
    assert terms[0] == "can i return an opened electronic product"
    # Phrase synonyms fired? "opened electronic" alone is no phrase but words
    # are token-expanded.
    assert "opened" in terms
    assert "open" in terms
    # 'electronics' should land via word-level synonym from 'electronic'.
    assert "electronics" in terms


def test_expand_query_phrase_synonyms_for_packaging() -> None:
    terms = expand_query("My product arrived with damaged packaging.")
    assert "opened box" in terms or "packaging damage" in terms


def test_expand_query_phrase_synonyms_for_delayed_flight() -> None:
    terms = expand_query("What happens if my flight is delayed more than 3 hours?")
    # "delayed flight" phrase triggers and produces airline-policy-friendly terms.
    assert any(
        t in terms for t in ("flight delay", "late departure", "disruption", "irrops")
    )


def test_expand_query_overage_phrase() -> None:
    terms = expand_query("Why was I charged overage?")
    assert "overage" in terms
    # Either the phrase synonym fired or a synonym word is present.
    assert any(
        t in terms for t in ("exceeded limit", "extra usage", "usage charge")
    )


def test_score_match_returns_metadata() -> None:
    terms = expand_query("opened electronic")
    score, fields, reason = score_match(
        terms,
        {
            "title": "Electronics — 14-day return",
            "body": "Opened items in resaleable condition are accepted with a 15% restocking fee.",
        },
    )
    assert score > 0
    assert "title" in fields or "body" in fields
    assert "matched" in reason


def test_score_match_no_hit() -> None:
    score, fields, reason = score_match(
        ["nothing matches here"],
        {"title": "Refund eligibility", "body": "Refunds within 30 days"},
    )
    assert score == 0.0
    assert fields == []
    assert "no direct" in reason or "no query" in reason


def test_make_excerpt_truncates() -> None:
    out = make_excerpt("a " * 500, max_len=80)
    assert out.endswith("…")
    assert len(out) <= 80


def test_infer_policy_types_for_common_phrases() -> None:
    assert "cancellation_policy" in infer_policy_types("What are the cancellation rules?")
    assert "baggage_policy" in infer_policy_types("baggage allowance for business class")
    assert "warranty_policy" in infer_policy_types("warranty exclusions on SKU-000001")
    assert "overage_policy" in infer_policy_types("Why was I charged overage?")


# ---------------------------------------------------------------------------
# Tool-level retrieval tests — the eight spec phrases
# ---------------------------------------------------------------------------


def test_opened_electronic_returns_return_rule(seeded_session: Session) -> None:
    """'opened electronic product' must find at least one commerce return rule."""
    out = search_return_rules.call(
        seeded_session, {"query": "opened electronic product"}
    )
    assert out["count"] >= 1, f"got zero rows: {out}"
    first = _ranked_first(out["rules"])
    _assert_match_fields(first)
    haystack = " ".join(
        [first["rule_name"], first["product_category_name"], first.get("body_excerpt", "")]
    ).lower()
    # Either the rule or category should be about electronics, or the body
    # should explicitly cover opened-item handling.
    assert any(
        kw in haystack
        for kw in ("electronics", "opened", "open", "device", "unsealed")
    ), first


def test_damaged_packaging_returns_relevant_text(seeded_session: Session) -> None:
    """'damaged packaging' should hit a return rule or a policy clause."""
    rules = search_return_rules.call(
        seeded_session, {"query": "damaged packaging"}
    )
    clauses = get_policy_clause.call(
        seeded_session, {"query": "damaged packaging"}
    )
    assert rules["count"] + clauses["count"] >= 1, (rules, clauses)
    if rules["count"]:
        _assert_match_fields(_ranked_first(rules["rules"]))
    if clauses["count"]:
        _assert_match_fields(_ranked_first(clauses["clauses"]))


def test_missing_accessories_returns_relevant_text(seeded_session: Session) -> None:
    rules = search_return_rules.call(
        seeded_session, {"query": "missing accessories"}
    )
    clauses = get_policy_clause.call(
        seeded_session, {"query": "missing accessories"}
    )
    assert rules["count"] + clauses["count"] >= 1, (rules, clauses)


def test_warranty_exclusions_returns_warranty_text(seeded_session: Session) -> None:
    """Policy clauses or warranty-policy documents should mention exclusions."""
    clauses = get_policy_clause.call(
        seeded_session, {"query": "warranty exclusions"}
    )
    docs = search_policy_documents.call(
        seeded_session, {"query": "warranty exclusions"}
    )
    assert clauses["count"] + docs["count"] >= 1, (clauses, docs)
    if clauses["count"]:
        first = _ranked_first(clauses["clauses"])
        _assert_match_fields(first)


def test_overage_charge_returns_saas_policy(seeded_session: Session) -> None:
    clauses = get_policy_clause.call(seeded_session, {"query": "overage charge"})
    docs = search_policy_documents.call(seeded_session, {"query": "overage charge"})
    assert clauses["count"] + docs["count"] >= 1, (clauses, docs)
    found = False
    for c in clauses["clauses"]:
        if c["policy_type"] == "overage_policy" or "overage" in c["body"].lower():
            found = True
            break
    for d in docs["documents"]:
        if d["policy_type"] == "overage_policy" or "overage" in d["excerpt"].lower():
            found = True
            break
    assert found, "no overage_policy match found"


def test_delayed_flight_returns_airline_policy(seeded_session: Session) -> None:
    """'flight delayed more than 3 hours' must surface airline policy text."""
    clauses = get_policy_clause.call(
        seeded_session, {"query": "flight delayed more than 3 hours"}
    )
    docs = search_policy_documents.call(
        seeded_session, {"query": "flight delayed more than 3 hours"}
    )
    assert clauses["count"] + docs["count"] >= 1, (clauses, docs)
    airline_hit = False
    for c in clauses["clauses"]:
        if c["policy_domain"] == "airline":
            airline_hit = True
            break
    for d in docs["documents"]:
        if d["domain"] == "airline":
            airline_hit = True
            break
    assert airline_hit, "no airline-domain match for delayed-flight query"


def test_cancellation_rules_returns_cancellation_policy(seeded_session: Session) -> None:
    """'cancellation rules' must surface cancellation_policy clauses."""
    clauses = get_policy_clause.call(
        seeded_session, {"query": "cancellation rules"}
    )
    docs = search_policy_documents.call(
        seeded_session, {"query": "cancellation rules"}
    )
    assert clauses["count"] + docs["count"] >= 1
    has_cancellation = any(
        c["policy_type"] == "cancellation_policy" for c in clauses["clauses"]
    ) or any(d["policy_type"] == "cancellation_policy" for d in docs["documents"])
    assert has_cancellation, "no cancellation_policy match found"


def test_baggage_allowance_returns_baggage_policy(seeded_session: Session) -> None:
    clauses = get_policy_clause.call(
        seeded_session, {"query": "baggage allowance"}
    )
    docs = search_policy_documents.call(
        seeded_session, {"query": "baggage allowance"}
    )
    assert clauses["count"] + docs["count"] >= 1
    has_baggage = any(
        c["policy_type"] == "baggage_policy" for c in clauses["clauses"]
    ) or any(d["policy_type"] == "baggage_policy" for d in docs["documents"])
    assert has_baggage, "no baggage_policy match found"


# ---------------------------------------------------------------------------
# Match-metadata contract — every text tool exposes the four fields
# ---------------------------------------------------------------------------


def test_search_return_rules_exposes_match_metadata(seeded_session: Session) -> None:
    out = search_return_rules.call(seeded_session, {"query": "return"})
    assert out["count"] >= 1
    _assert_match_fields(_ranked_first(out["rules"]))
    assert isinstance(out["query_terms"], list) and out["query_terms"]


def test_search_policy_documents_exposes_match_metadata(seeded_session: Session) -> None:
    out = search_policy_documents.call(seeded_session, {"query": "refund"})
    assert out["count"] >= 1
    _assert_match_fields(_ranked_first(out["documents"]))


def test_get_policy_clause_exposes_match_metadata(seeded_session: Session) -> None:
    out = get_policy_clause.call(seeded_session, {"query": "refund"})
    assert out["count"] >= 1
    _assert_match_fields(_ranked_first(out["clauses"]))


def test_get_product_warranty_terms_exposes_match_metadata(seeded_session: Session) -> None:
    from app.models import Product
    from sqlalchemy import select

    sku = seeded_session.execute(select(Product.sku).limit(1)).scalar_one()
    out = get_product_warranty_terms.call(seeded_session, {"sku": sku})
    assert out["count"] >= 1
    first = _ranked_first(out["terms"])
    _assert_match_fields(first)
    # The warranty-specific extra excerpt field.
    assert "exclusions_excerpt" in first


def test_search_internal_agent_notes_exposes_match_metadata(
    seeded_session: Session,
) -> None:
    from app.models import Customer
    from sqlalchemy import select

    cust = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    out = search_internal_agent_notes.call(
        seeded_session, {"customer_id": cust, "query": "vip"}
    )
    for n in out["notes"]:
        _assert_match_fields(n)


def test_search_operational_incidents_exposes_match_metadata(
    seeded_session: Session,
) -> None:
    out = search_operational_incidents.call(
        seeded_session, {"query": "delay", "domain": "airline"}
    )
    assert out["count"] >= 1
    _assert_match_fields(_ranked_first(out["incidents"]))


def test_get_support_resolution_template_exposes_match_metadata(
    seeded_session: Session,
) -> None:
    out = get_support_resolution_template.call(
        seeded_session, {"category": "refund_delay"}
    )
    assert out["count"] >= 1
    _assert_match_fields(_ranked_first(out["templates"]))


# ---------------------------------------------------------------------------
# Fallback behaviour — synthetic queries that don't appear in the seed text
# but should still resolve via the policy-type inference fallback
# ---------------------------------------------------------------------------


def test_fallback_kicks_in_when_direct_match_empty(seeded_session: Session) -> None:
    """'cancel and money back' is unlikely to appear verbatim, but the policy-
    type inference fallback should still surface cancellation_policy text."""
    out = get_policy_clause.call(
        seeded_session, {"query": "cancel and money back"}
    )
    # We don't strictly require fallback_used=True (a direct hit on 'cancel'
    # is also fine), but we must get *something*.
    assert out["count"] >= 1


def test_fallback_is_explicit_when_used(seeded_session: Session) -> None:
    """An unambiguous fallback case: a made-up phrase that still maps to a
    known policy_type via the inference table."""
    out = search_policy_documents.call(
        seeded_session, {"query": "extra usage exceeded my plan"}
    )
    assert out["count"] >= 1
    assert isinstance(out["inferred_policy_types"], list)
