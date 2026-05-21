"""Tests for the Phase 6B-6 text-knowledge eval cases.

Phase 6B-6 added a block of realistic policy/document/notes/incident questions
across commerce, airline, SaaS, support, CRM, and KB. These tests pin down:

  * the dataset grew by at least 300 cases (the spec floor);
  * each of the seven text-retrieval tools added in Phase 6B-4 appears in
    ``expected_tools`` for at least one new case;
  * the new cases use only registry tools and the right domains;
  * customer-voice cases without an id are flagged so the scorer doesn't
    penalise a chatbot that asks for clarification.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.tools import default_registry
from backend.scripts.generate_eval_cases import generate

PHASE_6B6_CATEGORIES = frozenset(
    {
        "text_returns_commerce",
        "text_warranty_generic",
        "text_warranty_sku",
        "text_airline_policy",
        "text_saas_policy",
        "text_support_sla",
        "text_support_template",
        "text_internal_notes",
        "text_incidents",
        "text_active_policy",
        "text_general_policy",
    }
)

# The seven text-retrieval tools added in Phase 6B-4 plus two bonuses.
TEXT_RETRIEVAL_TOOLS = frozenset(
    {
        "search_policy_documents",
        "get_policy_clause",
        "get_active_policy",
        "list_policy_versions",
        "search_return_rules",
        "get_product_warranty_terms",
        "search_internal_agent_notes",
        "search_operational_incidents",
        "get_support_resolution_template",
    }
)


@pytest.fixture(scope="session")
def generated_cases(seeded_engine) -> list[dict]:
    """One generation per session, against the shared seeded DB."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        return generate(s, seed=42)


@pytest.fixture(scope="session")
def phase_6b6_cases(generated_cases: list[dict]) -> list[dict]:
    return [c for c in generated_cases if c["category"] in PHASE_6B6_CATEGORIES]


# ---------------------------------------------------------------------------
# Volume + category coverage
# ---------------------------------------------------------------------------


def test_phase_6b6_added_at_least_300_cases(phase_6b6_cases: list[dict]) -> None:
    assert len(phase_6b6_cases) >= 300, (
        f"expected >=300 Phase 6B-6 cases, got {len(phase_6b6_cases)}"
    )


def test_phase_6b6_all_categories_present(phase_6b6_cases: list[dict]) -> None:
    seen = {c["category"] for c in phase_6b6_cases}
    missing = PHASE_6B6_CATEGORIES - seen
    assert not missing, f"missing categories: {missing}"


def test_phase_6b6_total_dataset_grew(generated_cases: list[dict]) -> None:
    # Pre-6B6 floor was 1008; 6B-6 adds >=300 so 1300+ is the new floor.
    assert len(generated_cases) >= 1300


# ---------------------------------------------------------------------------
# Tool coverage
# ---------------------------------------------------------------------------


def test_every_text_retrieval_tool_appears_in_at_least_one_case(
    phase_6b6_cases: list[dict],
) -> None:
    referenced: set[str] = set()
    for c in phase_6b6_cases:
        referenced.update(c["expected_tools"])
    missing = TEXT_RETRIEVAL_TOOLS - referenced
    assert not missing, (
        f"Phase 6B-6 must reference every new text tool — missing: {missing}"
    )


def test_phase_6b6_expected_tools_are_valid(phase_6b6_cases: list[dict]) -> None:
    registry = set(default_registry.names())
    for c in phase_6b6_cases:
        for tool in c["expected_tools"]:
            assert tool in registry, (
                f"case {c['id']} ({c['category']}) references unknown tool {tool!r}"
            )


def test_each_phase_6b6_case_has_at_least_one_expected_tool(
    phase_6b6_cases: list[dict],
) -> None:
    for c in phase_6b6_cases:
        # Customer-voice missing-context cases are still given candidate
        # tools (so clarification OR routing is both acceptable), so this
        # holds for every case in the new block.
        assert c["expected_tools"], (
            f"{c['id']} ({c['category']}) has empty expected_tools"
        )


def test_phase_6b6_domain_distribution(phase_6b6_cases: list[dict]) -> None:
    """All six target domains appear in the new cases."""
    domains = {c["expected_domain"] for c in phase_6b6_cases}
    # The block exercises commerce/airline/saas/support/crm/kb — every
    # one must be present.
    for d in ("commerce", "airline", "saas", "support", "crm", "kb"):
        assert d in domains, f"Phase 6B-6 missing domain: {d}"


# ---------------------------------------------------------------------------
# Fairness signals
# ---------------------------------------------------------------------------


def test_internal_notes_voice_cases_flag_missing_context(
    phase_6b6_cases: list[dict],
) -> None:
    """Customer-voice internal-note questions (no id, phrased as "my notes")
    must allow a clarification answer."""
    notes_voice = [
        c
        for c in phase_6b6_cases
        if c["category"] == "text_internal_notes" and c["customer_id"] is None
    ]
    # We expect a non-trivial number of these.
    assert notes_voice, "no missing-context internal_notes cases generated"
    for c in notes_voice:
        # Either the case is explicitly missing_context_expected, OR it carries
        # an external customer code in the message. Both are acceptable.
        if c["missing_context_expected"]:
            assert c["clarification_acceptable"] is True
            assert c["must_use_tool"] is False
        else:
            # External-id case — must_use_tool stays True.
            assert "CUST-" in c["message"], (
                f"{c['id']}: missing_context_expected=False but no external id in message"
            )


def test_phase_6b6_warranty_sku_cases_reference_warranty_tool(
    phase_6b6_cases: list[dict],
) -> None:
    sku_cases = [c for c in phase_6b6_cases if c["category"] == "text_warranty_sku"]
    assert sku_cases, "no SKU-specific warranty cases generated"
    for c in sku_cases:
        assert "get_product_warranty_terms" in c["expected_tools"]
        assert "SKU-" in c["message"], (
            f"{c['id']}: warranty SKU case missing SKU pattern in message"
        )


def test_phase_6b6_incident_cases_reference_incident_tool(
    phase_6b6_cases: list[dict],
) -> None:
    inc_cases = [c for c in phase_6b6_cases if c["category"] == "text_incidents"]
    assert inc_cases, "no incident cases generated"
    for c in inc_cases:
        assert "search_operational_incidents" in c["expected_tools"], c["id"]


def test_phase_6b6_template_cases_reference_template_tool(
    phase_6b6_cases: list[dict],
) -> None:
    tmpl_cases = [
        c for c in phase_6b6_cases if c["category"] == "text_support_template"
    ]
    assert tmpl_cases
    for c in tmpl_cases:
        assert "get_support_resolution_template" in c["expected_tools"], c["id"]


# ---------------------------------------------------------------------------
# Determinism + sequential ids preserved
# ---------------------------------------------------------------------------


def test_phase_6b6_ids_are_sequential(generated_cases: list[dict]) -> None:
    """Adding the new block must not break the global sequential id contract."""
    for idx, c in enumerate(generated_cases, start=1):
        assert c["id"] == f"eval_{idx:03d}"


def test_phase_6b6_generation_is_deterministic(seeded_engine) -> None:
    """Re-running the generator returns byte-identical Phase 6B-6 cases."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        a = [
            c
            for c in generate(s, seed=42)
            if c["category"] in PHASE_6B6_CATEGORIES
        ]
    with Session(engine) as s:
        b = [
            c
            for c in generate(s, seed=42)
            if c["category"] in PHASE_6B6_CATEGORIES
        ]
    assert a == b
