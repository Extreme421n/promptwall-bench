"""End-to-end tests for POST /chat (baseline mode + mock provider)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking, LLMCall, ToolInvocation, Trace


def _chat(api_client, **kwargs):
    body = {"mode": "baseline", "model": "mock"}
    body.update(kwargs)
    r = api_client.post("/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Tool-routing through the full loop
# ---------------------------------------------------------------------------


def test_chat_baggage_question_calls_baggage_tool(api_client) -> None:
    body = _chat(
        api_client,
        message="What's the checked baggage allowance on business class international flights?",
    )
    names = [tc["name"] for tc in body["tools_called"]]
    assert "get_baggage_policy" in names
    assert all(tc["success"] for tc in body["tools_called"])
    assert all(tc["evidence_id"] is not None for tc in body["tools_called"])
    # Answer should be grounded in the tool output (business / 32kg appear in seed).
    assert body["answer"]
    assert "business" in body["answer"].lower()


def test_chat_booking_question_with_pnr(api_client, seeded_session: Session) -> None:
    # Use a known PNR from the seed.
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    body = _chat(api_client, message=f"Can you pull up my booking {pnr}?")
    names = [tc["name"] for tc in body["tools_called"]]
    assert names == ["get_booking_details"]
    assert body["tools_called"][0]["arguments"] == {"booking_reference": pnr}
    assert pnr in body["answer"]


def test_chat_ambiguous_ticket_question_returns_clarification(api_client) -> None:
    """'Status of my ticket?' without a TKT- prefix or PNR is genuinely ambiguous.

    A fair baseline asks for clarification rather than guessing.
    """
    body = _chat(api_client, message="Status of my ticket please?")
    assert body["tools_called"] == []
    answer = body["answer"].lower()
    assert "flight ticket" in answer and "support ticket" in answer


def test_chat_support_ticket_with_prefix_calls_support_tool(api_client) -> None:
    body = _chat(api_client, message="Any update on TKT-WDR3VW?")
    names = [tc["name"] for tc in body["tools_called"]]
    assert names == ["get_support_ticket_status"]


def test_chat_refund_with_pnr_calls_refund_tool(api_client) -> None:
    body = _chat(api_client, message="Where is my refund for booking WDR3VW?")
    names = [tc["name"] for tc in body["tools_called"]]
    assert "get_refund_status" in names


def test_chat_flight_status_calls_flight_tool(api_client) -> None:
    body = _chat(api_client, message="What's the status of flight BA1234?")
    names = [tc["name"] for tc in body["tools_called"]]
    assert names == ["get_flight_status"]


def test_chat_how_to_question_calls_kb(api_client) -> None:
    body = _chat(api_client, message="How do I cancel a non-refundable fare?")
    names = [tc["name"] for tc in body["tools_called"]]
    assert "search_kb_articles" in names


def test_chat_completely_ambiguous_returns_clarification(api_client) -> None:
    body = _chat(api_client, message="Hi, can you help me?")
    assert body["tools_called"] == []
    assert body["evidence_ids"] == []
    assert body["answer"]


# ---------------------------------------------------------------------------
# Response shape + trace persistence
# ---------------------------------------------------------------------------


def test_chat_response_shape(api_client) -> None:
    body = _chat(api_client, message="What's the baggage allowance in economy?")
    # Every field declared in the contract is present.
    assert set(body) == {
        "answer",
        "trace_id",
        "session_id",
        "tools_called",
        "evidence_ids",
        "latency_ms",
        "estimated_cost_usd",
    }
    assert body["latency_ms"] >= 0
    assert isinstance(body["trace_id"], int)
    assert isinstance(body["session_id"], str)


def test_chat_persists_trace_with_llm_calls_and_tool_invocations(
    api_client, seeded_session: Session
) -> None:
    body = _chat(
        api_client,
        message="What's the baggage allowance on first class international flights?",
    )
    trace_id = body["trace_id"]

    # Trace exists and is closed.
    trace = seeded_session.execute(select(Trace).where(Trace.id == trace_id)).scalar_one()
    assert trace.user_message
    assert trace.final_answer == body["answer"]
    assert trace.ended_at is not None
    assert trace.latency_ms is not None
    assert trace.mode == "baseline"

    # At least two LLM calls were recorded (one for tool request, one for final).
    llm_calls = seeded_session.execute(
        select(LLMCall).where(LLMCall.trace_id == trace_id).order_by(LLMCall.id)
    ).scalars().all()
    assert len(llm_calls) >= 2
    assert llm_calls[0].tool_calls_requested is not None
    assert llm_calls[-1].tool_calls_requested is None  # final answer turn
    assert all(c.provider == "mock" for c in llm_calls)

    # One tool_invocations row matching the response.
    invs = seeded_session.execute(
        select(ToolInvocation).where(ToolInvocation.trace_id == trace_id)
    ).scalars().all()
    assert len(invs) == len(body["tools_called"]) == 1
    assert invs[0].tool_name == "get_baggage_policy"
    assert invs[0].evidence_id == body["evidence_ids"][0]


def test_chat_persists_no_tool_invocations_for_clarification(
    api_client, seeded_session: Session
) -> None:
    body = _chat(api_client, message="Hi there, can you help?")
    trace_id = body["trace_id"]

    invs = seeded_session.execute(
        select(ToolInvocation).where(ToolInvocation.trace_id == trace_id)
    ).scalars().all()
    assert invs == []

    llm_calls = seeded_session.execute(
        select(LLMCall).where(LLMCall.trace_id == trace_id)
    ).scalars().all()
    # Only one LLM call (final clarification, no tool round-trip).
    assert len(llm_calls) == 1
    assert llm_calls[0].tool_calls_requested is None
    assert llm_calls[0].output_message == body["answer"]


def test_chat_reuses_provided_session_id(api_client, seeded_session: Session) -> None:
    from app.models import ChatSession

    first = _chat(api_client, message="What's the baggage allowance in economy?")
    second = _chat(
        api_client,
        message="What about business class?",
        session_id=first["session_id"],
    )
    assert second["session_id"] == first["session_id"]

    # Both traces hang off the same chat_sessions row.
    chat = seeded_session.execute(
        select(ChatSession).where(ChatSession.session_uuid == first["session_id"])
    ).scalar_one()
    trace_session_ids = seeded_session.execute(
        select(Trace.session_id).where(
            Trace.id.in_([first["trace_id"], second["trace_id"]])
        )
    ).scalars().all()
    assert set(trace_session_ids) == {chat.id}


def test_chat_unsupported_model_returns_400(api_client) -> None:
    r = api_client.post(
        "/chat",
        json={"mode": "baseline", "model": "gpt-7", "message": "hi"},
    )
    assert r.status_code == 400


def test_chat_empty_message_rejected(api_client) -> None:
    r = api_client.post(
        "/chat",
        json={"mode": "baseline", "model": "mock", "message": ""},
    )
    assert r.status_code == 422
