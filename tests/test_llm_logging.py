"""Tests for TraceService.log_llm_call persistence."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm import ChatMessage, MockLLMProvider, messages_to_dicts
from app.models import LLMCall
from app.services import TraceService
from app.tools import default_registry


def test_log_llm_call_persists_row(seeded_session: Session) -> None:
    svc = TraceService(seeded_session)
    chat = svc.create_chat_session()
    trace = svc.create_trace(session_id=chat.id, user_message="hi")

    call = svc.log_llm_call(
        trace_id=trace.id,
        provider="mock",
        model="mock-1",
        input_messages=[{"role": "user", "content": "hi"}],
        output_message="hello there",
        tool_calls_requested=None,
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
        estimated_cost_usd=Decimal("0.000001"),
        latency_ms=12,
    )
    seeded_session.commit()

    fetched = seeded_session.execute(
        select(LLMCall).where(LLMCall.id == call.id)
    ).scalar_one()
    assert fetched.trace_id == trace.id
    assert fetched.provider == "mock"
    assert fetched.model == "mock-1"
    assert fetched.input_messages == [{"role": "user", "content": "hi"}]
    assert fetched.output_message == "hello there"
    assert fetched.prompt_tokens == 5
    assert fetched.completion_tokens == 3
    assert fetched.total_tokens == 8
    assert fetched.estimated_cost_usd == Decimal("0.000001")
    assert fetched.latency_ms == 12


def test_mock_provider_round_trip_logs_correctly(seeded_session: Session) -> None:
    """End-to-end: run the mock provider and persist the resulting LLM call."""
    svc = TraceService(seeded_session)
    chat = svc.create_chat_session()
    trace = svc.create_trace(
        session_id=chat.id, user_message="Where is my refund for booking AB12CD?"
    )

    provider = MockLLMProvider()
    messages = [ChatMessage(role="user", content="Where is my refund for booking AB12CD?")]
    resp = provider.chat(messages, tools=default_registry.describe_all())

    svc.log_llm_call(
        trace_id=trace.id,
        provider=resp.provider,
        model=resp.model,
        input_messages=messages_to_dicts(messages),
        output_message=resp.final_text,
        tool_calls_requested=[tc.model_dump(mode="json") for tc in resp.tool_calls],
        prompt_tokens=resp.token_usage.prompt_tokens,
        completion_tokens=resp.token_usage.completion_tokens,
        total_tokens=resp.token_usage.total_tokens,
        latency_ms=resp.latency_ms,
    )
    seeded_session.commit()

    persisted = seeded_session.execute(
        select(LLMCall).where(LLMCall.trace_id == trace.id)
    ).scalar_one()
    assert persisted.provider == "mock"
    assert persisted.tool_calls_requested is not None
    assert len(persisted.tool_calls_requested) == 1
    assert persisted.tool_calls_requested[0]["name"] == "get_refund_status"
    assert persisted.tool_calls_requested[0]["arguments"] == {
        "booking_reference": "AB12CD"
    }
    assert persisted.total_tokens > 0
