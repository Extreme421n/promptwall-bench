"""Tests for ToolExecutor: trace logging + evidence_id + error capture."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Customer, ToolInvocation, Trace
from app.services import ToolExecutor


def _count_invocations(session: Session) -> int:
    return session.execute(select(func.count()).select_from(ToolInvocation)).scalar_one()


def _count_traces(session: Session) -> int:
    return session.execute(select(func.count()).select_from(Trace)).scalar_one()


def test_execute_tool_logs_successful_invocation(seeded_session: Session) -> None:
    before_inv = _count_invocations(seeded_session)
    before_traces = _count_traces(seeded_session)

    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    executor = ToolExecutor(seeded_session)
    result = executor.execute_tool(
        trace_id=None,
        tool_name="get_customer_profile",
        input_json={"customer_id": cust_id},
    )

    assert result.success is True
    assert result.output is not None
    assert result.output["customer_id"] == cust_id
    assert result.evidence_id is not None
    assert result.evidence_id.startswith("ev_")
    assert result.latency_ms >= 0

    # One trace + one tool_invocation were inserted.
    assert _count_invocations(seeded_session) == before_inv + 1
    assert _count_traces(seeded_session) == before_traces + 1

    # The persisted row matches the result.
    persisted = seeded_session.execute(
        select(ToolInvocation).where(ToolInvocation.evidence_id == result.evidence_id)
    ).scalar_one()
    assert persisted.tool_name == "get_customer_profile"
    assert persisted.success is True
    assert persisted.input_json == {"customer_id": cust_id}
    assert persisted.output_json is not None
    assert persisted.error_message is None


def test_execute_tool_logs_failed_invocation(seeded_session: Session) -> None:
    before = _count_invocations(seeded_session)
    executor = ToolExecutor(seeded_session)
    result = executor.execute_tool(
        trace_id=None,
        tool_name="get_customer_profile",
        input_json={"customer_id": 99_999_999},  # not found
    )
    assert result.success is False
    assert result.error_type == "ResourceNotFoundError"
    assert result.evidence_id is None

    assert _count_invocations(seeded_session) == before + 1
    persisted = seeded_session.execute(
        select(ToolInvocation)
        .where(ToolInvocation.tool_name == "get_customer_profile")
        .order_by(ToolInvocation.id.desc())
        .limit(1)
    ).scalar_one()
    assert persisted.success is False
    assert persisted.output_json is None
    assert persisted.error_message is not None
    assert "not found" in persisted.error_message.lower()
    assert persisted.evidence_id is None


def test_execute_tool_logs_validation_error(seeded_session: Session) -> None:
    before = _count_invocations(seeded_session)
    executor = ToolExecutor(seeded_session)
    result = executor.execute_tool(
        trace_id=None,
        tool_name="get_customer_profile",
        input_json={},
    )
    assert result.success is False
    assert result.error_type == "ToolValidationError"
    assert _count_invocations(seeded_session) == before + 1


def test_execute_tool_logs_unknown_tool(seeded_session: Session) -> None:
    before = _count_invocations(seeded_session)
    executor = ToolExecutor(seeded_session)
    result = executor.execute_tool(
        trace_id=None,
        tool_name="not_a_tool",
        input_json={},
    )
    assert result.success is False
    assert result.error_type == "ToolNotFoundError"
    # We still log the failed lookup so it's visible in the trace.
    assert _count_invocations(seeded_session) == before + 1


def test_execute_tool_with_existing_trace(seeded_session: Session) -> None:
    from app.services import TraceService

    svc = TraceService(seeded_session)
    chat = svc.create_chat_session(channel="web")
    trace = svc.create_trace(
        session_id=chat.id, user_message="manual trace", mode="baseline"
    )
    seeded_session.commit()

    before_traces = _count_traces(seeded_session)

    executor = ToolExecutor(seeded_session)
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    result = executor.execute_tool(
        trace_id=trace.id,
        tool_name="get_customer_profile",
        input_json={"customer_id": cust_id},
    )
    assert result.success is True

    # No new trace was created — the executor reused the existing one.
    assert _count_traces(seeded_session) == before_traces

    # The invocation is linked to the supplied trace_id.
    persisted = seeded_session.execute(
        select(ToolInvocation).where(ToolInvocation.evidence_id == result.evidence_id)
    ).scalar_one()
    assert persisted.trace_id == trace.id


def test_execute_tool_evidence_ids_are_unique(seeded_session: Session) -> None:
    executor = ToolExecutor(seeded_session)
    cust_id = seeded_session.execute(select(Customer.id).limit(1)).scalar_one()
    a = executor.execute_tool(
        None, "get_customer_profile", {"customer_id": cust_id}
    )
    b = executor.execute_tool(
        None, "get_customer_profile", {"customer_id": cust_id}
    )
    assert a.success and b.success
    assert a.evidence_id != b.evidence_id
