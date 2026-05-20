"""Unit tests for TraceService."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ChatSession, ToolInvocation, Trace
from app.services import TraceService


def test_create_chat_session_and_trace(seeded_session: Session) -> None:
    svc = TraceService(seeded_session)
    chat = svc.create_chat_session(channel="web")
    assert chat.id is not None
    assert chat.session_uuid

    trace = svc.create_trace(
        session_id=chat.id,
        user_message="hello world",
        mode="baseline",
        metadata={"intent": "greeting"},
    )
    assert trace.id is not None
    assert trace.user_message == "hello world"
    assert trace.extra_metadata == {"intent": "greeting"}
    assert trace.started_at is not None
    assert trace.ended_at is None


def test_finish_trace_sets_ended_at_and_latency(seeded_session: Session) -> None:
    svc = TraceService(seeded_session)
    chat = svc.create_chat_session()
    trace = svc.create_trace(session_id=chat.id, user_message="q")
    finished = svc.finish_trace(trace.id, final_answer="a", latency_ms=42)
    assert finished.final_answer == "a"
    assert finished.latency_ms == 42
    assert finished.ended_at is not None


def test_finish_trace_computes_latency_when_not_provided(seeded_session: Session) -> None:
    svc = TraceService(seeded_session)
    chat = svc.create_chat_session()
    trace = svc.create_trace(session_id=chat.id, user_message="q")
    finished = svc.finish_trace(trace.id, final_answer="ok")
    assert finished.latency_ms is not None
    assert finished.latency_ms >= 0


def test_finish_trace_unknown_id_raises(seeded_session: Session) -> None:
    svc = TraceService(seeded_session)
    with pytest.raises(ValueError):
        svc.finish_trace(10_000_000, final_answer="x")


def test_log_tool_invocation_persists_row(seeded_session: Session) -> None:
    svc = TraceService(seeded_session)
    chat = svc.create_chat_session()
    trace = svc.create_trace(session_id=chat.id, user_message="q")

    inv = svc.log_tool_invocation(
        trace_id=trace.id,
        tool_name="get_customer_profile",
        input_json={"customer_id": 1},
        output_json={"customer_id": 1, "full_name": "x"},
        success=True,
        latency_ms=5,
        evidence_id="ev_abc123",
    )
    seeded_session.commit()

    fetched = seeded_session.execute(
        select(ToolInvocation).where(ToolInvocation.id == inv.id)
    ).scalar_one()
    assert fetched.tool_name == "get_customer_profile"
    assert fetched.success is True
    assert fetched.evidence_id == "ev_abc123"
    assert fetched.input_json == {"customer_id": 1}
    assert fetched.output_json == {"customer_id": 1, "full_name": "x"}
