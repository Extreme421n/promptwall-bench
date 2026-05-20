"""Tests for the PromptWall rule-based candidate analyzer (Phase 3A)."""

from __future__ import annotations

import pytest

from app.promptwall import CandidateDecision, PromptWallCandidateAnalyzer
from app.tools import default_registry


@pytest.fixture()
def analyzer() -> PromptWallCandidateAnalyzer:
    return PromptWallCandidateAnalyzer()


@pytest.fixture()
def all_tools() -> list[str]:
    return default_registry.names()


def _analyze(analyzer, msg, tools) -> CandidateDecision:
    return analyzer.analyze(message=msg, available_tools=tools)


# ---------------------------------------------------------------------------
# Greetings / no-tool
# ---------------------------------------------------------------------------


def test_greeting_predicts_no_tool(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "Hi, how are you?", all_tools)
    assert d.tool_required_predicted is False
    assert d.predicted_tools == []
    assert d.confidence >= 0.8
    assert "greeting" in d.reason.lower()


def test_empty_message_no_tool(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "   ", all_tools)
    assert d.tool_required_predicted is False
    assert d.predicted_tools == []


# ---------------------------------------------------------------------------
# Explicit identifiers route confidently
# ---------------------------------------------------------------------------


def test_ticket_prefix_predicts_support(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "Any update on TKT-WDR3VW?", all_tools)
    assert d.tool_required_predicted is True
    assert "get_support_ticket_status" in d.predicted_tools
    assert d.confidence >= 0.8


def test_booking_with_pnr_predicts_booking_details(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "Can you pull up booking WDR3VW?", all_tools)
    assert "get_booking_details" in d.predicted_tools
    assert d.tool_required_predicted is True


def test_flight_status_with_flight_number(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "What's the status of flight BA1234?", all_tools)
    assert "get_flight_status" in d.predicted_tools
    assert d.tool_required_predicted is True


def test_baggage_question_predicts_baggage_policy(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "What's the baggage allowance on business?", all_tools)
    assert "get_baggage_policy" in d.predicted_tools


def test_refund_with_pnr_predicts_refund_status(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "Where is my refund for booking AB12CD?", all_tools)
    assert "get_refund_status" in d.predicted_tools


# ---------------------------------------------------------------------------
# Ambiguous and policy framing
# ---------------------------------------------------------------------------


def test_lone_pnr_no_keyword_underdetermined(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "Hi, just checking on WDR3VW.", all_tools)
    # No 'booking'/'ticket' keyword → analyzer doesn't commit.
    assert d.tool_required_predicted is False
    assert d.predicted_tools == []


def test_how_to_question_predicts_kb_search(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "How do I cancel a non-refundable fare?", all_tools)
    assert "search_kb_articles" in d.predicted_tools
    assert d.tool_required_predicted is True


def test_policy_framing_predicts_policy_clause(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "What's your cancellation policy?", all_tools)
    assert "get_policy_clause" in d.predicted_tools


# ---------------------------------------------------------------------------
# New Phase 2E tools
# ---------------------------------------------------------------------------


def test_seat_availability_with_pnr(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "What seats are available on booking AB12CD?", all_tools)
    assert "search_available_seats" in d.predicted_tools


def test_change_fee_with_pnr(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "How much is the change fee for AB12CD?", all_tools)
    assert "calculate_change_fee" in d.predicted_tools


def test_loyalty_balance_with_email(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "What's my loyalty balance? Email: a@b.com", all_tools)
    assert "get_loyalty_balance" in d.predicted_tools
    assert "get_customer_profile" in d.predicted_tools


def test_open_issues_with_customer_id(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "Any open tickets for customer id 100?", all_tools)
    assert "get_customer_open_issues" in d.predicted_tools


def test_customer_search_by_name(analyzer, all_tools) -> None:
    d = _analyze(analyzer, "Find customer named Ada Lovelace", all_tools)
    assert "search_customer_records" in d.predicted_tools


# ---------------------------------------------------------------------------
# Output filtering
# ---------------------------------------------------------------------------


def test_predicted_tools_filtered_to_available(analyzer) -> None:
    """If a tool isn't in available_tools, it must not appear in predicted_tools."""
    d = analyzer.analyze(
        message="What's the baggage allowance in economy?",
        available_tools=["get_customer_profile"],  # baggage tool not exposed
    )
    assert "get_baggage_policy" not in d.predicted_tools


def test_confidence_in_unit_range(analyzer, all_tools) -> None:
    for msg in [
        "TKT-ABCDEF",
        "Hello",
        "How do I cancel a fare?",
        "Find customer named John Smith",
        "?",
    ]:
        d = _analyze(analyzer, msg, all_tools)
        assert 0.0 <= d.confidence <= 1.0
        assert d.reason  # non-empty rationale always present
