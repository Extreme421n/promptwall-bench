"""Tests for the eval-case generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.tools import default_registry
from backend.scripts.generate_eval_cases import (
    generate,
    main,
    write_jsonl,
)

REQUIRED_FIELDS = {
    "id",
    "category",
    "message",
    "expected_tools",
    "must_use_tool",
    "expected_domain",
    "risk",
    "notes",
    "customer_id",
    # Phase D1 — fairness signals
    "missing_context_expected",
    "clarification_acceptable",
}

ALLOWED_DOMAINS = {"airline", "support", "kb", "crm", "saas", "commerce"}
ALLOWED_RISK = {"low", "medium", "high"}


@pytest.fixture(scope="session")
def generated_cases(seeded_engine) -> list[dict]:
    """Run the generator once per session against the seeded DB."""
    engine, _ = seeded_engine
    with Session(engine) as s:
        return generate(s, seed=42)


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_generates_at_least_1000_cases(generated_cases: list[dict]) -> None:
    assert len(generated_cases) >= 1000


def test_all_cases_have_required_fields(generated_cases: list[dict]) -> None:
    for c in generated_cases:
        missing = REQUIRED_FIELDS - set(c)
        assert not missing, f"case {c.get('id')} missing fields: {missing}"


def test_ids_are_unique_and_sequential(generated_cases: list[dict]) -> None:
    ids = [c["id"] for c in generated_cases]
    assert len(set(ids)) == len(ids)
    # Sequential format eval_001, eval_002, ...
    for idx, case_id in enumerate(ids, start=1):
        assert case_id == f"eval_{idx:03d}"


def test_messages_non_empty(generated_cases: list[dict]) -> None:
    for c in generated_cases:
        assert isinstance(c["message"], str) and c["message"].strip()


def test_domains_and_risk_are_allowed(generated_cases: list[dict]) -> None:
    for c in generated_cases:
        assert c["expected_domain"] in ALLOWED_DOMAINS, c
        assert c["risk"] in ALLOWED_RISK, c


# ---------------------------------------------------------------------------
# Tool alignment with the live registry
# ---------------------------------------------------------------------------


def test_expected_tools_reference_only_registered_tools(generated_cases: list[dict]) -> None:
    registry_names = set(default_registry.names())
    for c in generated_cases:
        for tool in c["expected_tools"]:
            assert tool in registry_names, (
                f"case {c['id']} references unknown tool {tool!r}"
            )


def test_must_use_tool_implies_expected_tools_nonempty(
    generated_cases: list[dict],
) -> None:
    for c in generated_cases:
        if c["must_use_tool"]:
            assert c["expected_tools"], (
                f"case {c['id']} marked must_use_tool=True but has empty expected_tools"
            )


def test_mix_of_must_use_tool_true_and_false(generated_cases: list[dict]) -> None:
    truth = {c["must_use_tool"] for c in generated_cases}
    assert truth == {True, False}


# ---------------------------------------------------------------------------
# Category distribution
# ---------------------------------------------------------------------------


def test_all_phase_categories_present(generated_cases: list[dict]) -> None:
    expected_categories = {
        # Phase 2A
        "booking",
        "flight_status",
        "refund",
        "baggage",
        "support_ticket",
        "customer_loyalty",
        "kb_policy",
        "flight_search",
        "ambiguous",
        "missing_param",
        "no_tool",
        "adversarial",
        # Phase 2E
        "seat_availability",
        "change_fee",
        "change_options",
        "loyalty_balance",
        "policy_clause",
        "open_issues",
        "customer_search",
        # Phase D1 — SaaS
        "subscription_status",
        "plan_limits",
        "invoice_status",
        "usage_overage",
        "api_usage_summary",
        "saas_seat_alloc",
        # Phase D1 — Commerce
        "commerce_order_status",
        "commerce_refund_status",
        "shipment_status",
        "search_products",
        "product_details",
        "product_inventory",
        # Phase D1 — Multi-domain + extras
        "multi_domain_ambiguous",
        "missing_context_extra",
        # Phase C2 / D2 — new tool coverage
        "search_support_tickets",
        "escalation_policy",
        "create_ticket_draft",
        "search_policy_documents",
        "latest_policy_version",
        "calculate_bundle_price",
        "commerce_return_status",
        "customer_segment",
        # Phase D2 — multi-step
        "multi_step_refund_policy",
        "multi_step_flight_change",
        "multi_step_saas_overage_invoice",
        "multi_step_order_shipment",
        "multi_step_customer_issue",
        "multi_step_baggage_booking",
        "multi_step_segment_issues",
    }
    seen = {c["category"] for c in generated_cases}
    missing = expected_categories - seen
    assert not missing, f"missing categories: {missing}"


def test_all_six_domains_represented(generated_cases: list[dict]) -> None:
    expected_domains = {"airline", "support", "kb", "crm", "saas", "commerce"}
    seen = {c["expected_domain"] for c in generated_cases}
    missing = expected_domains - seen
    assert not missing, f"missing domains: {missing}"


def test_distribution_targets(generated_cases: list[dict]) -> None:
    """Check the dataset roughly matches the 65/20/10/5 target split.

    Buckets are derived from the category and the fairness flags:
      * ambiguous       = ambiguous + multi_domain_ambiguous + flight_search
      * missing_context = any case with missing_context_expected=True
      * adversarial     = category == 'adversarial'
      * normal          = everything else
    """
    total = len(generated_cases)
    ambiguous = sum(
        1
        for c in generated_cases
        if c["category"] in {"ambiguous", "multi_domain_ambiguous", "flight_search"}
    )
    missing = sum(1 for c in generated_cases if c["missing_context_expected"])
    adversarial = sum(1 for c in generated_cases if c["category"] == "adversarial")
    normal = total - ambiguous - missing - adversarial

    # Tolerant windows around the 65 / 20 / 10 / 5 targets.
    assert 0.55 <= normal / total <= 0.78, f"normal share {normal/total:.2%}"
    assert 0.10 <= ambiguous / total <= 0.28, f"ambiguous share {ambiguous/total:.2%}"
    assert 0.05 <= missing / total <= 0.18, f"missing share {missing/total:.2%}"
    assert adversarial / total <= 0.10, f"adversarial share {adversarial/total:.2%}"


def test_multi_step_cases_have_multiple_acceptable_tools(
    generated_cases: list[dict],
) -> None:
    """Phase D2 multi-step categories list ≥2 acceptable expected_tools so
    the scorer credits the chatbot for picking any sensible first step.
    """
    multi_step_categories = {
        "multi_step_refund_policy",
        "multi_step_flight_change",
        "multi_step_saas_overage_invoice",
        "multi_step_order_shipment",
        "multi_step_customer_issue",
        "multi_step_baggage_booking",
        "multi_step_segment_issues",
    }
    multi = [c for c in generated_cases if c["category"] in multi_step_categories]
    assert len(multi) >= 30, f"too few multi-step cases generated: {len(multi)}"
    for c in multi:
        assert len(c["expected_tools"]) >= 2, (
            f"{c['id']}: multi-step case has only {len(c['expected_tools'])} expected tools"
        )
        assert c["must_use_tool"] is True


def test_clarification_acceptable_implies_no_failure_intent(
    generated_cases: list[dict],
) -> None:
    """Cases marked clarification_acceptable=True should not be marked
    must_use_tool=True with non-empty expected_tools. If they are, the scorer
    will treat clarification as a failure — which is the bug we want to avoid.
    """
    for c in generated_cases:
        if c["clarification_acceptable"] and c["must_use_tool"]:
            assert not c["expected_tools"], (
                f"{c['id']}: clarification_acceptable=True but must_use_tool=True "
                "with required tools — this would penalise asking for clarification"
            )


def test_adversarial_is_a_minority(generated_cases: list[dict]) -> None:
    n_adv = sum(1 for c in generated_cases if c["category"] == "adversarial")
    # At most 15% of the dataset.
    assert n_adv > 0
    assert n_adv <= 0.15 * len(generated_cases)


def test_uses_real_ids_from_seed(generated_cases: list[dict], seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        from app.models import Booking, SupportTicket

        real_pnrs = set(
            s.execute(__import__("sqlalchemy").select(Booking.booking_reference)).scalars().all()
        )
        real_tickets = set(
            s.execute(__import__("sqlalchemy").select(SupportTicket.ticket_number)).scalars().all()
        )

    # Pull all bookings PNRs referenced in the booking category.
    booking_msgs = [c["message"] for c in generated_cases if c["category"] == "booking"]
    pnr_found = sum(any(pnr in m for pnr in real_pnrs) for m in booking_msgs)
    assert pnr_found >= len(booking_msgs) // 2, (
        "booking category should reference real PNRs"
    )

    ticket_msgs = [c["message"] for c in generated_cases if c["category"] == "support_ticket"]
    ticket_found = sum(any(tn in m for tn in real_tickets) for m in ticket_msgs)
    assert ticket_found >= len(ticket_msgs) // 2


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def test_write_jsonl_produces_valid_lines(
    tmp_path: Path, generated_cases: list[dict]
) -> None:
    out = tmp_path / "eval_cases.jsonl"
    write_jsonl(generated_cases, out)
    assert out.exists()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(generated_cases)
    for line in lines:
        row = json.loads(line)
        assert REQUIRED_FIELDS <= set(row)


def test_main_cli_writes_jsonl(tmp_path: Path, seeded_engine) -> None:
    engine, _ = seeded_engine
    db_url = str(engine.url)
    out = tmp_path / "out" / "eval_cases.jsonl"
    rc = main(["--db-url", db_url, "--output", str(out)])
    assert rc == 0
    assert out.exists()
    n_lines = sum(1 for _ in out.open())
    assert n_lines >= 150


def test_generator_is_deterministic(seeded_engine) -> None:
    engine, _ = seeded_engine
    with Session(engine) as s:
        a = generate(s, seed=42)
    with Session(engine) as s:
        b = generate(s, seed=42)
    assert a == b
