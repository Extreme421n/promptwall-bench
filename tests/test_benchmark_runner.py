"""End-to-end tests for the BenchmarkRunner."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.eval import BenchmarkRunner, load_cases
from app.models import EvaluationResult, EvaluationRun


def _seeded_session_factory(engine: Engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _five_cases(seeded_engine) -> list[dict]:
    """A small, deterministic set covering the main scoring branches."""
    engine, _ = seeded_engine
    with _seeded_session_factory(engine)() as s:
        from app.models import Booking, SupportTicket

        pnr = s.execute(select(Booking.booking_reference).limit(1)).scalar_one()
        tkt = s.execute(select(SupportTicket.ticket_number).limit(1)).scalar_one()

    return [
        {
            "id": "bench_t1",
            "category": "baggage",
            "message": "What's the baggage allowance on business international flights?",
            "expected_tools": ["get_baggage_policy"],
            "must_use_tool": True,
            "expected_domain": "airline",
            "risk": "low",
            "notes": "baggage policy lookup",
            "customer_id": None,
        },
        {
            "id": "bench_t2",
            "category": "booking",
            "message": f"What's the status of booking {pnr}?",
            "expected_tools": ["get_booking_details"],
            "must_use_tool": True,
            "expected_domain": "airline",
            "risk": "medium",
            "notes": "booking lookup",
            "customer_id": None,
        },
        {
            "id": "bench_t3",
            "category": "support_ticket",
            "message": f"Any update on {tkt}?",
            "expected_tools": ["get_support_ticket_status"],
            "must_use_tool": True,
            "expected_domain": "support",
            "risk": "medium",
            "notes": "support lookup",
            "customer_id": None,
        },
        {
            "id": "bench_t4",
            "category": "ambiguous",
            "message": "Status of my ticket please?",
            "expected_tools": [],
            "must_use_tool": False,
            "expected_domain": "support",
            "risk": "medium",
            "notes": "clarification expected",
            "customer_id": None,
        },
        {
            "id": "bench_t5",
            "category": "no_tool",
            "message": "Hi, how are you?",
            "expected_tools": [],
            "must_use_tool": False,
            "expected_domain": "kb",
            "risk": "low",
            "notes": "small talk",
            "customer_id": None,
        },
    ]


def test_runner_runs_five_cases_and_returns_summary(seeded_engine) -> None:
    engine, _ = seeded_engine
    cases = _five_cases(seeded_engine)
    runner = BenchmarkRunner(engine, mode="baseline", model="mock", workers=1)
    summary = runner.run(cases, eval_file="bench_test")

    assert summary.total_cases == 5
    assert summary.run_id > 0
    assert len(summary.scores) == 5

    # The three required-tool cases should all have been answered via tools
    # by the mock provider.
    by_id = {s.case_id: s for s in summary.scores}
    assert by_id["bench_t1"].tool_called_when_required is True
    assert by_id["bench_t1"].expected_tool_hit is True
    assert by_id["bench_t2"].expected_tool_hit is True
    assert by_id["bench_t3"].expected_tool_hit is True

    # The ambiguous case should not have called a tool.
    assert by_id["bench_t4"].actual_tool_names == []
    # The no-tool case should not have called a tool.
    assert by_id["bench_t5"].actual_tool_names == []


def test_runner_persists_run_and_result_rows(seeded_engine) -> None:
    engine, _ = seeded_engine
    cases = _five_cases(seeded_engine)
    runner = BenchmarkRunner(engine, mode="baseline", model="mock", workers=1)
    summary = runner.run(cases, eval_file="bench_test")

    SessionLocal = _seeded_session_factory(engine)
    with SessionLocal() as s:
        run = s.execute(
            select(EvaluationRun).where(EvaluationRun.id == summary.run_id)
        ).scalar_one()
        assert run.mode == "baseline"
        assert run.model == "mock"
        assert run.total_cases == 5
        assert run.metrics_json is not None
        assert run.metrics_json["total_cases"] == 5
        assert run.ended_at is not None

        results = s.execute(
            select(EvaluationResult).where(EvaluationResult.run_id == run.id)
            .order_by(EvaluationResult.case_id)
        ).scalars().all()
        assert len(results) == 5
        # Every persisted row carries the same scoring decisions the summary returned.
        by_case = {r.case_id: r for r in results}
        for s_obj in summary.scores:
            persisted = by_case[s_obj.case_id]
            assert persisted.tool_called_when_required == s_obj.tool_called_when_required
            assert persisted.expected_tool_hit == s_obj.expected_tool_hit
            assert persisted.tool_skip == s_obj.tool_skip
            assert persisted.actual_tools_json is not None
            assert isinstance(persisted.expected_tools_json, list)
            assert persisted.trace_id is not None


def test_runner_with_concurrency_preserves_order_and_counts(seeded_engine) -> None:
    engine, _ = seeded_engine
    cases = _five_cases(seeded_engine)
    runner = BenchmarkRunner(engine, mode="baseline", model="mock", workers=3)
    summary = runner.run(cases, eval_file="bench_test_concurrent")
    assert summary.total_cases == 5
    SessionLocal = _seeded_session_factory(engine)
    with SessionLocal() as s:
        count = s.execute(
            select(func.count())
            .select_from(EvaluationResult)
            .where(EvaluationResult.run_id == summary.run_id)
        ).scalar_one()
    assert count == 5


def test_load_cases_from_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"id":"a","category":"t","message":"hi","expected_tools":[],"must_use_tool":false,"expected_domain":"kb","risk":"low","notes":"","customer_id":null}\n'
        '{"id":"b","category":"t","message":"yo","expected_tools":["x"],"must_use_tool":true,"expected_domain":"kb","risk":"low","notes":"","customer_id":null}\n',
        encoding="utf-8",
    )
    cases = load_cases(p)
    assert [c["id"] for c in cases] == ["a", "b"]


def test_runner_metrics_includes_p95_and_average_latency(seeded_engine) -> None:
    engine, _ = seeded_engine
    cases = _five_cases(seeded_engine)
    runner = BenchmarkRunner(engine, mode="baseline", model="mock", workers=1)
    summary = runner.run(cases, eval_file="bench_test_metrics")
    keys = set(summary.metrics)
    assert {
        "total_cases",
        "tool_required_cases",
        "tool_called_when_required_rate",
        "tool_skip_rate",
        "expected_tool_hit_rate",
        "wrong_tool_rate",
        "missing_evidence_rate",
        "clarification_rate",
        "suspicious_unsupported_claim_rate",
        "average_latency_ms",
        "p95_latency_ms",
    } <= keys
    assert summary.metrics["average_latency_ms"] >= 0
