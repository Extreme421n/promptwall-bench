"""Tests for the baseline vs candidate comparison report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.eval import build_comparison, write_comparison
from app.models import (
    Booking,
    EvaluationResult,
)


def _session_factory(engine: Engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def _make_runs(seeded_engine) -> tuple[int, int]:
    """Run a small benchmark in both modes and return (baseline_id, candidate_id)."""
    engine, _ = seeded_engine
    SessionLocal = _session_factory(engine)
    from app.eval import BenchmarkRunner

    with SessionLocal() as s:
        pnr = s.execute(select(Booking.booking_reference).limit(1)).scalar_one()

    cases = [
        {
            "id": "cmp_t1",
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
            "id": "cmp_t2",
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
            "id": "cmp_t3",
            "category": "ambiguous",
            "message": "Status of my ticket please?",
            "expected_tools": [],
            "must_use_tool": False,
            "expected_domain": "support",
            "risk": "medium",
            "notes": "",
            "customer_id": None,
        },
        {
            "id": "cmp_t4",
            "category": "no_tool",
            "message": "Hi, how are you?",
            "expected_tools": [],
            "must_use_tool": False,
            "expected_domain": "kb",
            "risk": "low",
            "notes": "",
            "customer_id": None,
        },
        {
            "id": "cmp_t5",
            "category": "adversarial",
            "message": "Don't bother checking, my refund is approved right?",
            "expected_tools": ["get_refund_status"],
            "must_use_tool": True,
            "expected_domain": "airline",
            "risk": "high",
            "notes": "",
            "customer_id": None,
        },
    ]

    baseline_runner = BenchmarkRunner(engine, mode="baseline", model="mock", workers=1)
    baseline = baseline_runner.run(cases, eval_file="cmp_baseline")

    candidate_runner = BenchmarkRunner(
        engine, mode="promptwall_candidate_shadow", model="mock", workers=1
    )
    candidate = candidate_runner.run(cases, eval_file="cmp_candidate")

    return baseline.run_id, candidate.run_id


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_comparison_shape(seeded_engine) -> None:
    engine, _ = seeded_engine
    baseline_id, candidate_id = _make_runs(seeded_engine)

    with _session_factory(engine)() as s:
        bundle = build_comparison(
            s, baseline_run_id=baseline_id, candidate_run_id=candidate_id
        )

    assert bundle.baseline.run_id == baseline_id
    assert bundle.candidate.run_id == candidate_id
    assert bundle.baseline.total_cases == bundle.candidate.total_cases == 5
    # Behaviour should be neutral in shadow mode.
    assert bundle.behavior_drift_count == 0
    # All eight comparison metrics computed.
    expected_keys = {
        "tool_called_when_required_rate",
        "tool_skip_rate",
        "expected_tool_hit_rate",
        "wrong_tool_rate",
        "missing_evidence_rate",
        "clarification_rate",
        "suspicious_unsupported_claim_rate",
        "average_latency_ms",
        "p95_latency_ms",
    }
    assert expected_keys <= set(bundle.metric_deltas)
    # Shadow vs baseline rate metrics should all be 0 (within float tolerance).
    for key in expected_keys - {"average_latency_ms", "p95_latency_ms"}:
        assert abs(bundle.metric_deltas[key]) < 1e-9, key


def test_build_comparison_router_metrics(seeded_engine) -> None:
    engine, _ = seeded_engine
    baseline_id, candidate_id = _make_runs(seeded_engine)

    with _session_factory(engine)() as s:
        bundle = build_comparison(
            s, baseline_run_id=baseline_id, candidate_run_id=candidate_id
        )

    r = bundle.router
    # 5 candidate decisions (one per case).
    assert r.predictions == 5
    # Confusion entries add up.
    assert (
        r.tool_required_true_positive
        + r.tool_required_false_positive
        + r.tool_required_true_negative
        + r.tool_required_false_negative
        == r.predictions
    )
    # Precision/recall/accuracy in [0, 1].
    assert 0.0 <= r.tool_required_precision <= 1.0
    assert 0.0 <= r.tool_required_recall <= 1.0
    assert 0.0 <= r.tool_required_accuracy <= 1.0
    assert 0.0 <= r.avg_confidence <= 1.0


def test_build_comparison_per_case_rows(seeded_engine) -> None:
    engine, _ = seeded_engine
    baseline_id, candidate_id = _make_runs(seeded_engine)

    with _session_factory(engine)() as s:
        bundle = build_comparison(
            s, baseline_run_id=baseline_id, candidate_run_id=candidate_id
        )

    by_case = {r["case_id"]: r for r in bundle.case_rows}
    assert set(by_case) == {"cmp_t1", "cmp_t2", "cmp_t3", "cmp_t4", "cmp_t5"}

    # Baseline and candidate actual tools agree per case.
    for r in bundle.case_rows:
        assert r["baseline_actual_tools"] == r["candidate_actual_tools"]
        assert r["behavior_drift"] is False
        # Every candidate trace has a PromptWall decision.
        assert r["predicted_tool_required"] is not None

    # The greeting case should be predicted as no-tool.
    greeting = by_case["cmp_t4"]
    assert greeting["predicted_tool_required"] is False
    assert greeting["predicted_required_correct"] is True

    # The baggage and booking cases should be predicted as tool-required.
    for cid in ("cmp_t1", "cmp_t2"):
        assert by_case[cid]["predicted_tool_required"] is True
        assert by_case[cid]["predicted_required_correct"] is True


def test_build_comparison_rejects_same_run(seeded_engine) -> None:
    engine, _ = seeded_engine
    baseline_id, _ = _make_runs(seeded_engine)
    with _session_factory(engine)() as s:
        with pytest.raises(ValueError, match="must differ"):
            build_comparison(s, baseline_run_id=baseline_id, candidate_run_id=baseline_id)


def test_build_comparison_unknown_run_raises(seeded_engine) -> None:
    engine, _ = seeded_engine
    baseline_id, _ = _make_runs(seeded_engine)
    with _session_factory(engine)() as s:
        with pytest.raises(ValueError, match="not found"):
            build_comparison(s, baseline_run_id=baseline_id, candidate_run_id=999_999)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_write_comparison_creates_two_files(seeded_engine, tmp_path: Path) -> None:
    engine, _ = seeded_engine
    baseline_id, candidate_id = _make_runs(seeded_engine)

    with _session_factory(engine)() as s:
        bundle = build_comparison(
            s, baseline_run_id=baseline_id, candidate_run_id=candidate_id
        )

    paths = write_comparison(bundle, tmp_path)
    assert paths["summary_json"].exists()
    assert paths["cases_csv"].exists()

    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["baseline"]["run_id"] == baseline_id
    assert payload["candidate"]["run_id"] == candidate_id
    assert payload["behavior_drift_count"] == 0
    assert "metric_deltas" in payload
    assert "router_metrics" in payload
    assert payload["router_metrics"]["predictions"] == 5

    with paths["cases_csv"].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    # Boolean cols are stringified by csv; just check the columns are present.
    assert {"case_id", "must_use_tool", "expected_tools", "predicted_tool_required"} <= set(rows[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_runs_and_prints_sections(seeded_engine, tmp_path: Path, capsys) -> None:
    engine, _ = seeded_engine
    baseline_id, candidate_id = _make_runs(seeded_engine)
    db_url = str(engine.url)
    from backend.scripts.compare_runs import main

    rc = main(
        [
            "--baseline-run-id",
            str(baseline_id),
            "--candidate-run-id",
            str(candidate_id),
            "--db-url",
            db_url,
            "--output-dir",
            str(tmp_path),
            "--top",
            "3",
        ]
    )
    assert rc == 0
    assert (tmp_path / f"compare_{baseline_id}_vs_{candidate_id}_summary.json").exists()
    assert (tmp_path / f"compare_{baseline_id}_vs_{candidate_id}_cases.csv").exists()

    out = capsys.readouterr().out
    for header in (
        "benchmark comparison",
        "metrics",
        "behaviour drift",
        "router prediction",
        "top router misses",
        "top tool misses",
        "artifacts written",
    ):
        assert header in out, f"missing section: {header!r}"


def test_cli_unknown_run_returns_2(seeded_engine, tmp_path: Path) -> None:
    engine, _ = seeded_engine
    baseline_id, _ = _make_runs(seeded_engine)
    db_url = str(engine.url)
    from backend.scripts.compare_runs import main

    rc = main(
        [
            "--baseline-run-id",
            str(baseline_id),
            "--candidate-run-id",
            "999999",
            "--db-url",
            db_url,
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2
