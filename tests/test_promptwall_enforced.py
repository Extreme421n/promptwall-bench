"""Integration tests for the promptwall_enforced mode (Phase 4A)."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Booking,
    LLMCall,
    PromptWallCandidateDecision,
    ToolInvocation,
    Trace,
)


def _chat(api_client, **kwargs) -> dict:
    body = {"mode": "baseline", "model": "mock"}
    body.update(kwargs)
    r = api_client.post("/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Enforcement happens BEFORE the LLM and the evidence is injected
# ---------------------------------------------------------------------------


def test_enforced_mode_pre_executes_tool_and_reduces_llm_calls(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    message = f"What's the status of booking {pnr}?"

    baseline = _chat(api_client, message=message, mode="baseline")
    enforced = _chat(api_client, message=message, mode="promptwall_enforced")

    # In baseline the mock makes 2 LLM calls (tool-request then final answer).
    # In enforced mode PromptWall pre-executes the tool and the mock only
    # needs one LLM call to synthesise the final answer.
    seeded_session.expire_all()
    baseline_llm = seeded_session.execute(
        select(func.count())
        .select_from(LLMCall)
        .where(LLMCall.trace_id == baseline["trace_id"])
    ).scalar_one()
    enforced_llm = seeded_session.execute(
        select(func.count())
        .select_from(LLMCall)
        .where(LLMCall.trace_id == enforced["trace_id"])
    ).scalar_one()
    assert baseline_llm >= 2
    assert enforced_llm == 1, (
        f"expected exactly one LLM call in enforced mode, got {enforced_llm}"
    )

    # The forced invocation is the only one persisted for the enforced trace.
    invs = seeded_session.execute(
        select(ToolInvocation)
        .where(ToolInvocation.trace_id == enforced["trace_id"])
        .order_by(ToolInvocation.id)
    ).scalars().all()
    assert len(invs) == 1
    assert invs[0].tool_name == "get_booking_details"
    assert invs[0].success is True
    assert invs[0].evidence_id is not None

    # The evidence_id appears in the enforced /chat response.
    assert invs[0].evidence_id in enforced["evidence_ids"]

    # The forced tool_call appears in tools_called.
    assert any(t["name"] == "get_booking_details" for t in enforced["tools_called"])


def test_enforced_mode_injects_evidence_into_llm_input(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    body = _chat(
        api_client,
        message=f"What's the status of booking {pnr}?",
        mode="promptwall_enforced",
    )
    seeded_session.expire_all()
    llm = seeded_session.execute(
        select(LLMCall).where(LLMCall.trace_id == body["trace_id"]).limit(1)
    ).scalar_one()

    # Build a single string of the input messages and look for the evidence
    # markers + the pre-completed tool turn.
    blob = " ".join(
        (m.get("content") or "") for m in (llm.input_messages or [])
    )
    assert "PromptWall verified evidence" in blob
    assert "evidence_id:" in blob

    # The final assistant turn is a tool_calls request and is followed by a
    # tool result with the booking payload.
    roles = [m["role"] for m in llm.input_messages]
    assert "tool" in roles
    tool_msg = next(m for m in llm.input_messages if m["role"] == "tool")
    assert pnr in tool_msg.get("content", "")


def test_enforced_mode_does_not_offer_forced_tool_to_llm(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    body = _chat(
        api_client,
        message=f"What's the status of booking {pnr}?",
        mode="promptwall_enforced",
    )
    # If the forced tool were re-offered, the mock provider would attempt to
    # call it again and we'd see more than one tool_invocation. Already
    # asserted above (==1) but pin the contract here too.
    invs = seeded_session.execute(
        select(func.count())
        .select_from(ToolInvocation)
        .where(ToolInvocation.trace_id == body["trace_id"])
    ).scalar_one()
    assert invs == 1


def test_enforced_mode_records_candidate_decision(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    body = _chat(
        api_client,
        message=f"What's the status of booking {pnr}?",
        mode="promptwall_enforced",
    )
    seeded_session.expire_all()
    decision = seeded_session.execute(
        select(PromptWallCandidateDecision).where(
            PromptWallCandidateDecision.trace_id == body["trace_id"]
        )
    ).scalar_one()
    assert decision.tool_required_predicted is True
    assert "get_booking_details" in decision.predicted_tools


def test_enforced_mode_trace_mode_is_recorded(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    body = _chat(
        api_client,
        message=f"What's the status of booking {pnr}?",
        mode="promptwall_enforced",
    )
    seeded_session.expire_all()
    trace = seeded_session.execute(
        select(Trace).where(Trace.id == body["trace_id"])
    ).scalar_one()
    assert trace.mode == "promptwall_enforced"


def test_enforced_answer_grounded_in_evidence(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    body = _chat(
        api_client,
        message=f"What's the status of booking {pnr}?",
        mode="promptwall_enforced",
    )
    assert pnr in body["answer"]


# ---------------------------------------------------------------------------
# Low-confidence cases must NOT enforce — behaviour should match baseline
# ---------------------------------------------------------------------------


def test_low_confidence_message_falls_back_to_baseline_behavior(
    api_client, seeded_session: Session
) -> None:
    """'Status of my ticket?' is genuinely ambiguous; no enforcement.

    Behaviour and final answer should look like baseline mode.
    """
    baseline = _chat(api_client, message="Status of my ticket please?", mode="baseline")
    enforced = _chat(
        api_client, message="Status of my ticket please?", mode="promptwall_enforced"
    )
    assert enforced["tools_called"] == baseline["tools_called"] == []
    assert enforced["answer"] == baseline["answer"]

    # No tool invocation should exist for the enforced trace.
    invs = seeded_session.execute(
        select(func.count())
        .select_from(ToolInvocation)
        .where(ToolInvocation.trace_id == enforced["trace_id"])
    ).scalar_one()
    assert invs == 0


def test_lone_pnr_no_intent_does_not_enforce(api_client, seeded_session: Session) -> None:
    """A bare 6-char code with no booking/refund/flight keyword must not be enforced."""
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    enforced = _chat(
        api_client, message=f"Hi, just checking on {pnr}.", mode="promptwall_enforced"
    )
    assert enforced["tools_called"] == []


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_runs_in_enforced_mode(seeded_engine) -> None:
    from app.eval import BenchmarkRunner

    engine, _ = seeded_engine
    pnr = _first_pnr(engine)

    cases = [
        # High-confidence enforced cases
        {
            "id": "pw_enf_1",
            "category": "booking",
            "message": f"What's the status of booking {pnr}?",
            "expected_tools": ["get_booking_details"],
            "must_use_tool": True,
            "expected_domain": "airline",
            "risk": "medium",
            "notes": "",
            "customer_id": None,
        },
        {
            "id": "pw_enf_2",
            "category": "baggage",
            "message": "Baggage allowance for business international flights?",
            "expected_tools": ["get_baggage_policy"],
            "must_use_tool": True,
            "expected_domain": "airline",
            "risk": "low",
            "notes": "",
            "customer_id": None,
        },
        # Low-confidence case (falls back)
        {
            "id": "pw_enf_3",
            "category": "ambiguous",
            "message": "Status of my ticket please?",
            "expected_tools": [],
            "must_use_tool": False,
            "expected_domain": "support",
            "risk": "medium",
            "notes": "",
            "customer_id": None,
        },
        # No-tool small talk
        {
            "id": "pw_enf_4",
            "category": "no_tool",
            "message": "Hi, how are you?",
            "expected_tools": [],
            "must_use_tool": False,
            "expected_domain": "kb",
            "risk": "low",
            "notes": "",
            "customer_id": None,
        },
    ]
    runner = BenchmarkRunner(engine, mode="promptwall_enforced", model="mock", workers=1)
    summary = runner.run(cases, eval_file="enforce_smoke")
    assert summary.total_cases == 4

    by_id = {s.case_id: s for s in summary.scores}
    # High-confidence cases used the right tool.
    assert by_id["pw_enf_1"].expected_tool_hit is True
    assert by_id["pw_enf_2"].expected_tool_hit is True
    # Low-confidence cases did not force a tool.
    assert by_id["pw_enf_3"].actual_tool_names == []
    assert by_id["pw_enf_4"].actual_tool_names == []


def _first_pnr(engine) -> str:
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        return s.execute(select(Booking.booking_reference).limit(1)).scalar_one()
