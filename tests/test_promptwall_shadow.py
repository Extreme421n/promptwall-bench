"""Integration tests for the promptwall_candidate_shadow mode.

These verify two things together:
1. The chat *behaviour* is unchanged from baseline (same tools called, same
   final answer for the same question).
2. A row is written to ``promptwall_candidate_decisions`` linked to the trace.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    Booking,
    PromptWallCandidateDecision,
    Trace,
)


def _chat(api_client, **kwargs) -> dict:
    body = {"mode": "baseline", "model": "mock"}
    body.update(kwargs)
    r = api_client.post("/chat", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_shadow_mode_does_not_change_answer_or_tools(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()

    baseline = _chat(api_client, message=f"What's the status of booking {pnr}?", mode="baseline")
    shadow = _chat(
        api_client,
        message=f"What's the status of booking {pnr}?",
        mode="promptwall_candidate_shadow",
    )

    assert baseline["tools_called"] == shadow["tools_called"] or (
        # Tool names + arguments match, even if evidence_ids differ each run.
        [(t["name"], t["arguments"]) for t in baseline["tools_called"]]
        == [(t["name"], t["arguments"]) for t in shadow["tools_called"]]
    )
    assert baseline["answer"] == shadow["answer"]


def test_shadow_mode_logs_candidate_decision(
    api_client, seeded_session: Session
) -> None:
    pnr = seeded_session.execute(select(Booking.booking_reference).limit(1)).scalar_one()
    body = _chat(
        api_client,
        message=f"What's the status of booking {pnr}?",
        mode="promptwall_candidate_shadow",
    )
    trace_id = body["trace_id"]

    seeded_session.expire_all()
    trace = seeded_session.execute(
        select(Trace).where(Trace.id == trace_id)
    ).scalar_one()
    assert trace.mode == "promptwall_candidate_shadow"

    decision = seeded_session.execute(
        select(PromptWallCandidateDecision).where(
            PromptWallCandidateDecision.trace_id == trace_id
        )
    ).scalar_one()
    assert decision.tool_required_predicted is True
    assert "get_booking_details" in decision.predicted_tools
    assert 0.0 < decision.confidence <= 1.0
    assert decision.reason


def test_baseline_mode_does_not_log_candidate_decision(
    api_client, seeded_session: Session
) -> None:
    body = _chat(api_client, message="What's the status of booking AB12CD?", mode="baseline")
    trace_id = body["trace_id"]
    seeded_session.expire_all()
    n = seeded_session.execute(
        select(func.count())
        .select_from(PromptWallCandidateDecision)
        .where(PromptWallCandidateDecision.trace_id == trace_id)
    ).scalar_one()
    assert n == 0


def test_shadow_mode_greeting_predicts_no_tool(api_client, seeded_session: Session) -> None:
    body = _chat(api_client, message="Hi, how are you?", mode="promptwall_candidate_shadow")
    decision = seeded_session.execute(
        select(PromptWallCandidateDecision).where(
            PromptWallCandidateDecision.trace_id == body["trace_id"]
        )
    ).scalar_one()
    assert decision.tool_required_predicted is False
    assert decision.predicted_tools == []


def test_benchmark_runs_in_shadow_mode(seeded_engine) -> None:
    """The benchmark runner must accept the new mode and produce a summary."""
    engine, _ = seeded_engine
    from app.eval import BenchmarkRunner

    # A small handful of cases is enough to exercise the path.
    pnr_seq = seeded_session_pnrs(engine)
    cases = [
        {
            "id": "shadow_t1",
            "category": "baggage",
            "message": "What's the baggage allowance on business international flights?",
            "expected_tools": ["get_baggage_policy"],
            "must_use_tool": True,
            "expected_domain": "airline",
            "risk": "low",
            "notes": "",
            "customer_id": None,
        },
        {
            "id": "shadow_t2",
            "category": "booking",
            "message": f"What's the status of booking {pnr_seq[0]}?",
            "expected_tools": ["get_booking_details"],
            "must_use_tool": True,
            "expected_domain": "airline",
            "risk": "medium",
            "notes": "",
            "customer_id": None,
        },
        {
            "id": "shadow_t3",
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
    runner = BenchmarkRunner(
        engine, mode="promptwall_candidate_shadow", model="mock", workers=1
    )
    summary = runner.run(cases, eval_file="shadow_smoke")
    assert summary.total_cases == 3

    # Confirm a candidate decision was written for each trace produced.
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    with SessionLocal() as s:
        decisions = s.execute(
            select(PromptWallCandidateDecision)
            .join(Trace, Trace.id == PromptWallCandidateDecision.trace_id)
            .where(Trace.mode == "promptwall_candidate_shadow")
        ).scalars().all()
    assert len(decisions) >= 3


def seeded_session_pnrs(engine) -> list[str]:
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        return list(
            s.execute(select(Booking.booking_reference).limit(5)).scalars().all()
        )
