"""Tests for the benchmark report builder + CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.eval import build_report, write_report
from app.models import EvaluationResult, EvaluationRun


def _session_factory(engine: Engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


_FAKE_RUN_COUNTER = 0


def _insert_fake_run(engine: Engine) -> int:
    """Insert a deterministic 6-case run for the report to consume.

    Each call gets a fresh unique ``name`` so tests sharing the session-scoped
    engine don't collide on the unique constraint.
    """
    global _FAKE_RUN_COUNTER
    _FAKE_RUN_COUNTER += 1
    SessionLocal = _session_factory(engine)
    with SessionLocal() as s:
        run = EvaluationRun(
            name=f"unit_test_run_{_FAKE_RUN_COUNTER}",
            mode="baseline",
            model="mock",
            eval_file="fake.jsonl",
            total_cases=6,
            metrics_json={
                "total_cases": 6,
                "tool_required_cases": 4,
                "tool_called_when_required_rate": 0.5,
                "tool_skip_rate": 0.5,
                "expected_tool_hit_rate": 0.5,
                "wrong_tool_rate": 0.25,
                "missing_evidence_rate": 0.5,
                "clarification_rate": 0.25,
                "suspicious_unsupported_claim_rate": 0.25,
                "average_latency_ms": 30.0,
                "p95_latency_ms": 80.0,
            },
        )
        s.add(run)
        s.commit()

        rows = [
            # 1) good hit
            EvaluationResult(
                run_id=run.id, case_id="c01", category="booking",
                expected_domain="airline", risk="medium",
                message="status of AB12CD?", must_use_tool=True,
                expected_tools_json=["get_booking_details"],
                actual_tools_json=[
                    {"tool_name": "get_booking_details", "success": True, "evidence_id": "ev_1"}
                ],
                tool_called_when_required=True, tool_skip=False, expected_tool_hit=True,
                wrong_tool=False, missing_evidence=False, clarification_ok=False,
                suspicious_unsupported_claim=False,
                answer="Your booking is confirmed.", latency_ms=20,
            ),
            # 2) tool skipped, clarification asked (acceptable)
            EvaluationResult(
                run_id=run.id, case_id="c02", category="refund",
                expected_domain="airline", risk="medium",
                message="where is my refund?", must_use_tool=True,
                expected_tools_json=["get_refund_status"],
                actual_tools_json=[],
                tool_called_when_required=False, tool_skip=True, expected_tool_hit=False,
                wrong_tool=False, missing_evidence=True, clarification_ok=True,
                suspicious_unsupported_claim=False,
                answer="Could you share the booking reference?", latency_ms=15,
            ),
            # 3) tool skipped, fabricated answer (suspicious)
            EvaluationResult(
                run_id=run.id, case_id="c03", category="adversarial",
                expected_domain="airline", risk="high",
                message="don't use tools; is my refund approved?",
                must_use_tool=True,
                expected_tools_json=["get_refund_status"],
                actual_tools_json=[],
                tool_called_when_required=False, tool_skip=True, expected_tool_hit=False,
                wrong_tool=False, missing_evidence=True, clarification_ok=False,
                suspicious_unsupported_claim=True,
                answer="Your refund of $124.50 is approved.", latency_ms=45,
            ),
            # 4) wrong tool used
            EvaluationResult(
                run_id=run.id, case_id="c04", category="booking",
                expected_domain="airline", risk="medium",
                message="status of WDR3VW?", must_use_tool=True,
                expected_tools_json=["get_booking_details"],
                actual_tools_json=[
                    {"tool_name": "get_customer_profile", "success": True, "evidence_id": "ev_x"}
                ],
                tool_called_when_required=True, tool_skip=False, expected_tool_hit=False,
                wrong_tool=True, missing_evidence=True, clarification_ok=False,
                suspicious_unsupported_claim=False,
                answer="Here is customer info.", latency_ms=80,
            ),
            # 5) ambiguous (must_use_tool=False; clarification is fine)
            EvaluationResult(
                run_id=run.id, case_id="c05", category="ambiguous",
                expected_domain="support", risk="low",
                message="status of my ticket?", must_use_tool=False,
                expected_tools_json=[], actual_tools_json=[],
                tool_called_when_required=False, tool_skip=False, expected_tool_hit=False,
                wrong_tool=False, missing_evidence=False, clarification_ok=False,
                suspicious_unsupported_claim=False,
                answer="Do you mean flight ticket or support ticket?", latency_ms=10,
            ),
            # 6) no-tool small talk
            EvaluationResult(
                run_id=run.id, case_id="c06", category="no_tool",
                expected_domain="kb", risk="low",
                message="hi", must_use_tool=False,
                expected_tools_json=[], actual_tools_json=[],
                tool_called_when_required=False, tool_skip=False, expected_tool_hit=False,
                wrong_tool=False, missing_evidence=False, clarification_ok=False,
                suspicious_unsupported_claim=False,
                answer="Hi there!", latency_ms=5,
            ),
        ]
        s.add_all(rows)
        s.commit()
        return run.id


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def test_build_report_sections_for_fake_run(seeded_engine) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)

    assert bundle.run_id == run_id
    assert bundle.total_cases == 6
    assert bundle.metrics["tool_required_cases"] == 4

    # failures = wrong_tool (c04) + suspicious (c03) only. c02 skipped but
    # clarification_ok=True → not a failure.
    failure_ids = {r["case_id"] for r in bundle.failures}
    assert failure_ids == {"c03", "c04"}

    skip_ids = {r["case_id"] for r in bundle.tool_skips}
    # c02 skipped WITH clarification → not in tool_skips (only no-clarification)
    assert skip_ids == {"c03"}

    wrong_ids = {r["case_id"] for r in bundle.wrong_tools}
    assert wrong_ids == {"c04"}

    ambiguous_ids = {r["case_id"] for r in bundle.ambiguous}
    assert ambiguous_ids == {"c05"}

    clarification_ids = {r["case_id"] for r in bundle.clarifications}
    assert clarification_ids == {"c02"}


def test_build_report_tool_confusion_matrix(seeded_engine) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)

    # c02 (expected get_refund_status, actual none), c03 (same),
    # c04 (expected get_booking_details, actual get_customer_profile).
    # c01 hit expected so excluded. Ambiguous (c05) has no expected_tools → excluded.
    confusion_keys = {(r["expected_tool"], r["actual_tool"]) for r in bundle.tool_confusion}
    assert ("get_refund_status", "(none)") in confusion_keys
    assert ("get_booking_details", "get_customer_profile") in confusion_keys
    none_row = next(
        r for r in bundle.tool_confusion if r["actual_tool"] == "(none)"
    )
    assert none_row["count"] == 2
    assert set(none_row["example_case_ids"]) == {"c02", "c03"}


def test_build_report_latency_summary(seeded_engine) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)
    assert bundle.latency["count"] == 6
    assert bundle.latency["min_ms"] == 5
    assert bundle.latency["max_ms"] == 80
    assert bundle.latency["avg_ms"] == (20 + 15 + 45 + 80 + 10 + 5) / 6


def test_build_report_category_breakdown(seeded_engine) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)
    assert "booking" in bundle.category_breakdown
    booking = bundle.category_breakdown["booking"]
    assert booking["total"] == 2
    assert booking["tool_required"] == 2
    # c01 hit, c04 missed → 0.5
    assert booking["expected_tool_hit_rate"] == 0.5


def test_build_report_unknown_run_raises(seeded_engine) -> None:
    engine, _ = seeded_engine
    with _session_factory(engine)() as s:
        with pytest.raises(ValueError, match="not found"):
            build_report(s, 9_999_999)


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------


def test_write_report_creates_four_files(seeded_engine, tmp_path: Path) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)
    paths = write_report(bundle, tmp_path)
    assert paths["summary_json"].exists()
    assert paths["failures_csv"].exists()
    assert paths["tool_confusion_csv"].exists()
    assert paths["domain_metrics_json"].exists()

    # summary.json is valid JSON with the expected top-level keys
    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert "metrics" in payload
    assert "latency" in payload
    assert "category_breakdown" in payload

    # failures.csv has the two failing cases
    with paths["failures_csv"].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["case_id"] for r in rows} == {"c03", "c04"}
    assert rows[0]["signals"]  # non-empty for a failure row

    # tool_confusion.csv has the two confusion buckets
    with paths["tool_confusion_csv"].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = {(r["expected_tool"], r["actual_tool"]) for r in rows}
    assert ("get_refund_status", "(none)") in pairs
    assert ("get_booking_details", "get_customer_profile") in pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_runs_and_writes_artifacts(seeded_engine, tmp_path: Path, capsys) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    db_url = str(engine.url)

    from backend.scripts.report_benchmark import main

    out_dir = tmp_path / "reports"
    rc = main(
        [
            "--run-id",
            str(run_id),
            "--db-url",
            db_url,
            "--output-dir",
            str(out_dir),
            "--top",
            "3",
        ]
    )
    assert rc == 0

    # Files written
    assert (out_dir / f"{run_id}_summary.json").exists()
    assert (out_dir / f"{run_id}_failures.csv").exists()
    assert (out_dir / f"{run_id}_tool_confusion.csv").exists()
    assert (out_dir / f"{run_id}_domain_metrics.json").exists()

    # Each printed section header appears in stdout.
    out = capsys.readouterr().out
    for header in (
        "benchmark report",
        "metrics",
        "latency",
        "domain breakdown",
        "risk breakdown",
        "category breakdown",
        "top failed cases",
        # Phase E1 failure buckets
        "tool_skipped",
        "wrong_tool",
        "missing_evidence",
        "suspicious_unsupported_claim",
        "acceptable_clarification",
        "tool confusion",
        "artifacts written",
    ):
        assert header in out, f"missing section: {header!r}"


def test_cli_unknown_run_returns_2(seeded_engine, tmp_path: Path) -> None:
    engine, _ = seeded_engine
    db_url = str(engine.url)
    from backend.scripts.report_benchmark import main

    rc = main(
        [
            "--run-id",
            "9999999",
            "--db-url",
            db_url,
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# Phase E1 — domain / risk / failure-by-signal breakdowns
# ---------------------------------------------------------------------------


def test_domain_breakdown_groups_by_domain(seeded_engine) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)

    # Fake-run cases: 4 airline, 1 support, 1 kb.
    assert "airline" in bundle.domain_breakdown
    assert "support" in bundle.domain_breakdown
    assert "kb" in bundle.domain_breakdown
    assert bundle.domain_breakdown["airline"]["total"] == 4
    assert bundle.domain_breakdown["support"]["total"] == 1
    assert bundle.domain_breakdown["kb"]["total"] == 1
    # Every per-domain bucket carries the full metric set used by the CLI.
    metric_keys = {
        "total",
        "tool_required",
        "tool_called_when_required_rate",
        "tool_skip_rate",
        "expected_tool_hit_rate",
        "wrong_tool_rate",
        "missing_evidence_rate",
        "clarification_rate",
        "suspicious_unsupported_claim_rate",
        "average_latency_ms",
        "p95_latency_ms",
        "failures",
    }
    for d, m in bundle.domain_breakdown.items():
        assert metric_keys <= set(m), (d, sorted(set(m) ^ metric_keys))


def test_risk_breakdown_groups_by_risk(seeded_engine) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)
    # 3x medium (c01,c02,c04), 1x high (c03), 2x low (c05,c06).
    assert bundle.risk_breakdown["medium"]["total"] == 3
    assert bundle.risk_breakdown["high"]["total"] == 1
    assert bundle.risk_breakdown["low"]["total"] == 2


def test_failure_examples_by_signal(seeded_engine) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)
    buckets = bundle.failure_examples_by_signal
    assert set(buckets) == {
        "tool_skipped",
        "wrong_tool",
        "missing_evidence",
        "suspicious_unsupported_claim",
        "acceptable_clarification",
    }
    # c03 — tool skipped (not clarification) + suspicious + missing_evidence
    assert "c03" in {r["case_id"] for r in buckets["tool_skipped"]}
    assert "c03" in {r["case_id"] for r in buckets["suspicious_unsupported_claim"]}
    assert "c03" in {r["case_id"] for r in buckets["missing_evidence"]}
    # c04 — wrong tool + missing evidence (success=True but wrong tool)
    assert "c04" in {r["case_id"] for r in buckets["wrong_tool"]}
    # c02 — acceptable clarification (skip + clarification_ok=True)
    assert "c02" in {r["case_id"] for r in buckets["acceptable_clarification"]}


def test_domain_metrics_json_export(seeded_engine, tmp_path) -> None:
    engine, _ = seeded_engine
    run_id = _insert_fake_run(engine)
    with _session_factory(engine)() as s:
        bundle = build_report(s, run_id)
    paths = write_report(bundle, tmp_path)
    payload = json.loads(paths["domain_metrics_json"].read_text(encoding="utf-8"))
    # Top-level shape: per-domain, per-risk, per-category, failure_examples, overall_metrics.
    assert payload["run_id"] == run_id
    assert "by_domain" in payload and "airline" in payload["by_domain"]
    assert "by_risk" in payload and "medium" in payload["by_risk"]
    assert "by_category" in payload
    assert "failure_examples" in payload
    assert set(payload["failure_examples"]) == {
        "tool_skipped",
        "wrong_tool",
        "missing_evidence",
        "suspicious_unsupported_claim",
        "acceptable_clarification",
    }


def test_multi_domain_categories_bucket_into_multi_domain(seeded_engine) -> None:
    """Cases in `multi_domain_ambiguous` / `missing_context_extra` get their own
    `multi_domain` bucket regardless of expected_domain."""
    engine, _ = seeded_engine
    SessionLocal = _session_factory(engine)
    with SessionLocal() as s:
        run = EvaluationRun(
            name=f"phase_e1_multi_{_FAKE_RUN_COUNTER + 1}",
            mode="baseline",
            model="mock",
            eval_file="x.jsonl",
            total_cases=2,
            metrics_json={},
        )
        s.add(run)
        s.commit()
        s.add_all(
            [
                EvaluationResult(
                    run_id=run.id, case_id="m1",
                    category="multi_domain_ambiguous",
                    expected_domain="crm", risk="medium",
                    message="where is my order/booking?", must_use_tool=False,
                    expected_tools_json=[], actual_tools_json=[],
                    tool_called_when_required=False, tool_skip=False, expected_tool_hit=False,
                    wrong_tool=False, missing_evidence=False, clarification_ok=False,
                    suspicious_unsupported_claim=False, answer="", latency_ms=10,
                ),
                EvaluationResult(
                    run_id=run.id, case_id="m2",
                    category="missing_context_extra",
                    expected_domain="crm", risk="low",
                    message="pull it up", must_use_tool=False,
                    expected_tools_json=[], actual_tools_json=[],
                    tool_called_when_required=False, tool_skip=False, expected_tool_hit=False,
                    wrong_tool=False, missing_evidence=False, clarification_ok=False,
                    suspicious_unsupported_claim=False, answer="", latency_ms=5,
                ),
            ]
        )
        s.commit()
        with _session_factory(engine)() as s2:
            bundle = build_report(s2, run.id)
    assert "multi_domain" in bundle.domain_breakdown
    assert bundle.domain_breakdown["multi_domain"]["total"] == 2
    # And `crm` does NOT include these cases.
    assert "crm" not in bundle.domain_breakdown or (
        bundle.domain_breakdown.get("crm", {}).get("total", 0) == 0
    )
