"""Tests for the MockLLMProvider heuristic behaviour."""

from __future__ import annotations

import json

import pytest

from app.llm import ChatMessage, LLMResponse, LLMToolCall, MockLLMProvider
from app.tools import default_registry


@pytest.fixture()
def tool_specs() -> list[dict]:
    """All 8 registered tools, as the chatbot will pass them in."""
    return default_registry.describe_all()


@pytest.fixture()
def provider() -> MockLLMProvider:
    return MockLLMProvider()


def _user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def _only_tool_call(resp: LLMResponse) -> LLMToolCall:
    assert resp.final_text is None, f"expected tool call, got text: {resp.final_text!r}"
    assert len(resp.tool_calls) == 1, f"expected 1 tool call, got {len(resp.tool_calls)}"
    return resp.tool_calls[0]


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------


def test_baggage_question_calls_baggage_policy(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat(
        [_user("What's the checked baggage allowance on business class?")],
        tools=tool_specs,
    )
    call = _only_tool_call(resp)
    assert call.name == "get_baggage_policy"
    assert call.arguments == {"cabin_class": "business"}


def test_baggage_with_route_type(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat(
        [_user("How much luggage am I allowed on international flights in economy?")],
        tools=tool_specs,
    )
    call = _only_tool_call(resp)
    assert call.name == "get_baggage_policy"
    assert call.arguments["cabin_class"] == "economy"
    assert call.arguments["route_type"] == "international"


def test_refund_with_pnr_calls_refund_status(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat(
        [_user("Where is my refund for booking AB12CD?")],
        tools=tool_specs,
    )
    call = _only_tool_call(resp)
    assert call.name == "get_refund_status"
    assert call.arguments == {"booking_reference": "AB12CD"}


def test_refund_without_pnr_asks_for_clarification(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat(
        [_user("My refund hasn't arrived, can you help?")],
        tools=tool_specs,
    )
    assert resp.tool_calls == []
    assert resp.final_text is not None
    assert "booking reference" in resp.final_text.lower()


def test_flight_status_with_flight_number(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat(
        [_user("What's the flight status of BA178?")],
        tools=tool_specs,
    )
    call = _only_tool_call(resp)
    assert call.name == "get_flight_status"
    assert call.arguments == {"flight_number": "BA178"}


def test_support_ticket_prefix_routes_to_support_tool(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat(
        [_user("Any update on TKT-AB12CD?")],
        tools=tool_specs,
    )
    call = _only_tool_call(resp)
    assert call.name == "get_support_ticket_status"
    assert call.arguments == {"ticket_number": "TKT-AB12CD"}


def test_booking_question_with_pnr(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat(
        [_user("Can you pull up my booking WDR3VW?")],
        tools=tool_specs,
    )
    call = _only_tool_call(resp)
    assert call.name == "get_booking_details"
    assert call.arguments == {"booking_reference": "WDR3VW"}


def test_ticket_without_prefix_asks_for_clarification(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    """'ticket' alone (no TKT- prefix, no PNR) is ambiguous: flight vs support."""
    resp = provider.chat(
        [_user("Status of my ticket please?")],
        tools=tool_specs,
    )
    assert resp.tool_calls == []
    assert resp.final_text is not None
    assert (
        "flight ticket" in resp.final_text.lower()
        and "support ticket" in resp.final_text.lower()
    )


def test_lone_pnr_in_vague_message_asks_for_clarification(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat([_user("Hi, just checking on WDR3VW.")], tools=tool_specs)
    assert resp.tool_calls == []
    assert resp.final_text is not None


def test_completely_ambiguous_question_returns_clarification(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat([_user("Hi, can you help me?")], tools=tool_specs)
    assert resp.tool_calls == []
    assert resp.final_text is not None
    assert "more detail" in resp.final_text.lower() or "booking" in resp.final_text.lower()


def test_kb_policy_question(provider: MockLLMProvider, tool_specs: list[dict]) -> None:
    resp = provider.chat(
        [_user("How do I cancel a non-refundable fare?")],
        tools=tool_specs,
    )
    call = _only_tool_call(resp)
    assert call.name == "search_kb_articles"
    assert "query" in call.arguments


# ---------------------------------------------------------------------------
# Followup: synthesize answer from tool result
# ---------------------------------------------------------------------------


def test_followup_answer_from_booking_details_tool_result(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    user = _user("Can you pull up my booking WDR3VW?")
    first = provider.chat([user], tools=tool_specs)
    call = _only_tool_call(first)

    fake_result = {
        "count": 1,
        "bookings": [
            {
                "booking_reference": "WDR3VW",
                "customer_id": 97,
                "customer_name": "Deborah Rios",
                "flight_number": "EK2687",
                "booking_status": "cancelled",
                "cabin_class": "economy",
                "total_paid": "255.00",
                "currency": "USD",
                "scheduled_departure": "2026-05-29T22:45:00",
                "scheduled_arrival": "2026-05-30T01:00:00",
            }
        ],
    }
    second = provider.chat(
        [
            user,
            ChatMessage(role="assistant", tool_calls=[call]),
            ChatMessage(
                role="tool",
                tool_call_id=call.id,
                name=call.name,
                content=json.dumps(fake_result),
            ),
        ],
        tools=tool_specs,
    )
    assert second.tool_calls == []
    assert second.final_text is not None
    assert "WDR3VW" in second.final_text
    assert "cancelled" in second.final_text


def test_response_metadata_populated(
    provider: MockLLMProvider, tool_specs: list[dict]
) -> None:
    resp = provider.chat([_user("baggage on first class?")], tools=tool_specs)
    assert resp.provider == "mock"
    assert resp.model == "mock-1"
    assert resp.latency_ms >= 0
    assert resp.token_usage.total_tokens > 0
    assert resp.raw_response.get("mock") is True


def test_provider_respects_available_tools_only(provider: MockLLMProvider) -> None:
    """If the registry doesn't expose a tool, the mock must not call it."""
    # Offer only the customer profile tool.
    only_customer = [
        t for t in default_registry.describe_all() if t["name"] == "get_customer_profile"
    ]
    resp = provider.chat(
        [_user("What is my baggage allowance in business class?")],
        tools=only_customer,
    )
    # No matching tool offered → clarification
    assert resp.tool_calls == []
    assert resp.final_text is not None
