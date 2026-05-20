"""Unit tests for the PromptWall enforcement router (Phase 4A)."""

from __future__ import annotations

import pytest

from app.promptwall import EnforcementDecision, PromptWallRouter
from app.tools import default_registry


@pytest.fixture()
def all_tools() -> set[str]:
    return set(default_registry.names())


@pytest.fixture()
def router() -> PromptWallRouter:
    return PromptWallRouter()


def _decide(router, msg, tools) -> EnforcementDecision:
    return router.decide(message=msg, available_tools=tools)


# ---------------------------------------------------------------------------
# Positive matches — should enforce
# ---------------------------------------------------------------------------


def test_ticket_prefix_enforces_support(router, all_tools) -> None:
    d = _decide(router, "Any update on TKT-WDR3VW?", all_tools)
    assert d.should_enforce is True
    assert d.tool_name == "get_support_ticket_status"
    assert d.arguments == {"ticket_number": "TKT-WDR3VW"}
    assert d.confidence >= 0.85


def test_refund_with_pnr_enforces_refund_status(router, all_tools) -> None:
    d = _decide(router, "Where is my refund for booking AB12CD?", all_tools)
    assert d.should_enforce is True
    assert d.tool_name == "get_refund_status"
    assert d.arguments == {"booking_reference": "AB12CD"}


def test_booking_status_with_pnr_enforces_booking_details(router, all_tools) -> None:
    d = _decide(router, "What's the status of booking AB12CD?", all_tools)
    assert d.should_enforce is True
    assert d.tool_name == "get_booking_details"
    assert d.arguments == {"booking_reference": "AB12CD"}


def test_flight_status_with_flight_number_enforces_flight_status(router, all_tools) -> None:
    d = _decide(router, "Has flight BA1234 departed yet?", all_tools)
    assert d.should_enforce is True
    assert d.tool_name == "get_flight_status"
    assert d.arguments == {"flight_number": "BA1234"}


def test_baggage_with_cabin_enforces_baggage_policy(router, all_tools) -> None:
    d = _decide(router, "What's the baggage allowance on business class?", all_tools)
    assert d.should_enforce is True
    assert d.tool_name == "get_baggage_policy"
    assert d.arguments["cabin_class"] == "business"


def test_baggage_with_cabin_and_route(router, all_tools) -> None:
    d = _decide(
        router,
        "How much luggage am I allowed on international flights in economy?",
        all_tools,
    )
    assert d.should_enforce is True
    assert d.arguments == {"cabin_class": "economy", "route_type": "international"}


# ---------------------------------------------------------------------------
# Negative matches — should NOT enforce
# ---------------------------------------------------------------------------


def test_baggage_without_cabin_does_not_enforce(router, all_tools) -> None:
    """No detectable cabin class → can't fill the required argument."""
    d = _decide(router, "What's the baggage allowance?", all_tools)
    assert d.should_enforce is False


def test_refund_without_pnr_does_not_enforce(router, all_tools) -> None:
    d = _decide(router, "My refund hasn't arrived yet — can you help?", all_tools)
    assert d.should_enforce is False


def test_non_refundable_keyword_does_not_misfire_refund(router, all_tools) -> None:
    d = _decide(router, "How do I cancel a non-refundable fare?", all_tools)
    # Even with no PNR, the substring "refund" appears. The router must not
    # treat "non-refundable" as a refund-status intent.
    assert d.should_enforce is False


def test_ambiguous_ticket_does_not_enforce(router, all_tools) -> None:
    d = _decide(router, "Status of my ticket please?", all_tools)
    assert d.should_enforce is False


def test_completely_ambiguous_does_not_enforce(router, all_tools) -> None:
    d = _decide(router, "Hi, can you help me?", all_tools)
    assert d.should_enforce is False


def test_lone_pnr_in_vague_message_does_not_enforce(router, all_tools) -> None:
    """No booking/refund/flight intent → no enforcement."""
    d = _decide(router, "Hi, just checking on WDR3VW.", all_tools)
    assert d.should_enforce is False


def test_empty_message_does_not_enforce(router, all_tools) -> None:
    d = _decide(router, "   ", all_tools)
    assert d.should_enforce is False


# ---------------------------------------------------------------------------
# Threshold and availability gating
# ---------------------------------------------------------------------------


def test_threshold_gate_blocks_low_confidence(all_tools) -> None:
    """A threshold above every rule's confidence forces no enforcement."""
    router = PromptWallRouter(confidence_threshold=0.99)
    d = router.decide(message="What's the status of booking AB12CD?", available_tools=all_tools)
    assert d.should_enforce is False


def test_unavailable_tool_disables_rule(router) -> None:
    """If the relevant tool isn't exposed, the router must not enforce."""
    # No get_booking_details in the offered set.
    d = router.decide(
        message="What's the status of booking AB12CD?",
        available_tools={"get_baggage_policy"},
    )
    assert d.should_enforce is False
