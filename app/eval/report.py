"""Build human + machine-readable reports from an evaluation_run.

This module is pure: it queries the DB and produces dataclasses; the CLI
formats them. Tests can drive it without touching stdout.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import EvaluationResult, EvaluationRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    return float(sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f))


def _safe_rate(n: int, d: int) -> float:
    return float(n) / d if d else 0.0


def _truncate(text: Optional[str], n: int = 160) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _actual_tool_names(result: EvaluationResult) -> list[str]:
    return [t.get("tool_name") for t in (result.actual_tools_json or []) if t.get("tool_name")]


def _is_failure(r: EvaluationResult) -> bool:
    """A 'failure' fires any of the badness signals."""
    if r.wrong_tool:
        return True
    if r.suspicious_unsupported_claim:
        return True
    if r.must_use_tool and r.tool_skip and not r.clarification_ok:
        return True
    if r.must_use_tool and r.missing_evidence and not r.clarification_ok:
        return True
    return False


def _failure_signals(r: EvaluationResult) -> list[str]:
    signals: list[str] = []
    if r.tool_skip and not r.clarification_ok and r.must_use_tool:
        signals.append("tool_skip")
    if r.wrong_tool:
        signals.append("wrong_tool")
    if r.missing_evidence and r.must_use_tool and not r.clarification_ok:
        signals.append("missing_evidence")
    if r.suspicious_unsupported_claim:
        signals.append("suspicious_unsupported_claim")
    return signals


# ---------------------------------------------------------------------------
# Report bundle
# ---------------------------------------------------------------------------


@dataclass
class ReportBundle:
    run_id: int
    name: str
    mode: str
    model: str
    eval_file: Optional[str]
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    total_cases: int

    metrics: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    category_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Phase E1
    domain_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_examples_by_signal: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Lists of rows (dicts) — each list is a printed section + drives the CSVs.
    failures: list[dict[str, Any]] = field(default_factory=list)
    tool_skips: list[dict[str, Any]] = field(default_factory=list)
    wrong_tools: list[dict[str, Any]] = field(default_factory=list)
    ambiguous: list[dict[str, Any]] = field(default_factory=list)
    clarifications: list[dict[str, Any]] = field(default_factory=list)
    tool_confusion: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_report(session: Session, run_id: int) -> ReportBundle:
    run = session.execute(
        select(EvaluationRun).where(EvaluationRun.id == run_id)
    ).scalar_one_or_none()
    if run is None:
        raise ValueError(f"evaluation run {run_id} not found")

    results: list[EvaluationResult] = list(
        session.execute(
            select(EvaluationResult)
            .where(EvaluationResult.run_id == run_id)
            .order_by(EvaluationResult.case_id)
        )
        .scalars()
        .all()
    )

    bundle = ReportBundle(
        run_id=run.id,
        name=run.name,
        mode=run.mode,
        model=run.model,
        eval_file=run.eval_file,
        started_at=run.started_at,
        ended_at=run.ended_at,
        total_cases=run.total_cases,
        metrics=dict(run.metrics_json or {}),
    )

    bundle.latency = _latency_summary(results)
    bundle.category_breakdown = _category_breakdown(results)
    bundle.domain_breakdown = _domain_breakdown(results)
    bundle.risk_breakdown = _risk_breakdown(results)

    bundle.failures = _failure_rows(results)
    bundle.tool_skips = _tool_skip_rows(results)
    bundle.wrong_tools = _wrong_tool_rows(results)
    bundle.ambiguous = _ambiguous_rows(results)
    bundle.clarifications = _clarification_rows(results)
    bundle.tool_confusion = _tool_confusion_rows(results)
    bundle.failure_examples_by_signal = _failure_examples_by_signal(results)

    return bundle


def _latency_summary(results: list[EvaluationResult]) -> dict[str, float]:
    lats = [r.latency_ms for r in results if r.latency_ms is not None]
    if not lats:
        return {"count": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "count": len(lats),
        "avg_ms": sum(lats) / len(lats),
        "min_ms": float(min(lats)),
        "max_ms": float(max(lats)),
        "p50_ms": _percentile(lats, 50),
        "p95_ms": _percentile(lats, 95),
        "p99_ms": _percentile(lats, 99),
    }


def _metrics_for_subset(rows: list[EvaluationResult]) -> dict[str, Any]:
    """Compute the standard scorer metrics over an arbitrary subset of rows.

    Reused by category/domain/risk breakdowns so every slice reports the
    same fields with the same denominators.
    """
    tool_required = [r for r in rows if r.must_use_tool]
    any_called = [r for r in rows if r.actual_tools_json]
    expected_present = [r for r in rows if r.expected_tools_json]
    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    return {
        "total": len(rows),
        "tool_required": len(tool_required),
        "tool_called_when_required_rate": _safe_rate(
            sum(1 for r in tool_required if r.tool_called_when_required),
            len(tool_required),
        ),
        "tool_skip_rate": _safe_rate(
            sum(1 for r in tool_required if r.tool_skip),
            len(tool_required),
        ),
        "expected_tool_hit_rate": _safe_rate(
            sum(1 for r in expected_present if r.expected_tool_hit),
            len(expected_present),
        ),
        "wrong_tool_rate": _safe_rate(
            sum(1 for r in any_called if r.wrong_tool),
            len(any_called),
        ),
        "missing_evidence_rate": _safe_rate(
            sum(1 for r in tool_required if r.missing_evidence),
            len(tool_required),
        ),
        "clarification_rate": _safe_rate(
            sum(1 for r in tool_required if r.clarification_ok),
            len(tool_required),
        ),
        "suspicious_unsupported_claim_rate": _safe_rate(
            sum(1 for r in tool_required if r.suspicious_unsupported_claim),
            len(tool_required),
        ),
        "average_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 95) if latencies else 0.0,
        "failures": sum(1 for r in rows if _is_failure(r)),
    }


def _category_breakdown(results: list[EvaluationResult]) -> dict[str, dict[str, Any]]:
    by_cat: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    return {cat: _metrics_for_subset(rows) for cat, rows in by_cat.items()}


# ---------------------------------------------------------------------------
# Phase E1 — domain + risk slicing
# ---------------------------------------------------------------------------

# Phase D1's multi-domain ambiguous category gets its own bucket so it doesn't
# get washed into the dominant `crm` domain it's stored under.
_MULTI_DOMAIN_CATEGORIES = {"multi_domain_ambiguous", "missing_context_extra"}


def _domain_for(r: EvaluationResult) -> str:
    if r.category in _MULTI_DOMAIN_CATEGORIES:
        return "multi_domain"
    return r.expected_domain or "unknown"


def _domain_breakdown(results: list[EvaluationResult]) -> dict[str, dict[str, Any]]:
    by_domain: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        by_domain[_domain_for(r)].append(r)
    return {d: _metrics_for_subset(rows) for d, rows in by_domain.items()}


def _risk_breakdown(results: list[EvaluationResult]) -> dict[str, dict[str, Any]]:
    by_risk: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        by_risk[r.risk or "unknown"].append(r)
    return {risk: _metrics_for_subset(rows) for risk, rows in by_risk.items()}


# ---------------------------------------------------------------------------
# Phase E1 — top failure examples bucketed by signal
# ---------------------------------------------------------------------------


def _failure_examples_by_signal(
    results: list[EvaluationResult], *, top_per_bucket: int = 10
) -> dict[str, list[dict[str, Any]]]:
    """Group cases into the five diagnostic buckets and return up to N each.

    Four of the buckets are failures; ``acceptable_clarification`` is a
    *positive* signal — included alongside so reviewers can see the
    chatbot doing the right thing on hard cases.
    """
    buckets: dict[str, list[EvaluationResult]] = {
        "tool_skipped": [],
        "wrong_tool": [],
        "missing_evidence": [],
        "suspicious_unsupported_claim": [],
        "acceptable_clarification": [],
    }
    for r in results:
        if r.tool_skip and r.must_use_tool and not r.clarification_ok:
            buckets["tool_skipped"].append(r)
        if r.wrong_tool:
            buckets["wrong_tool"].append(r)
        if r.missing_evidence and r.must_use_tool and not r.clarification_ok:
            buckets["missing_evidence"].append(r)
        if r.suspicious_unsupported_claim:
            buckets["suspicious_unsupported_claim"].append(r)
        if r.clarification_ok:
            buckets["acceptable_clarification"].append(r)

    return {
        name: [_row_dict(r) for r in rows[:top_per_bucket]]
        for name, rows in buckets.items()
    }


def _row_dict(r: EvaluationResult, *, signals: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "case_id": r.case_id,
        "category": r.category,
        "must_use_tool": r.must_use_tool,
        "expected_tools": list(r.expected_tools_json or []),
        "actual_tools": _actual_tool_names(r),
        "tool_skip": r.tool_skip,
        "wrong_tool": r.wrong_tool,
        "missing_evidence": r.missing_evidence,
        "clarification_ok": r.clarification_ok,
        "suspicious_unsupported_claim": r.suspicious_unsupported_claim,
        "signals": signals if signals is not None else _failure_signals(r),
        "latency_ms": r.latency_ms,
        "message": r.message,
        "answer": r.answer or "",
        "trace_id": r.trace_id,
    }


def _failure_rows(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    return [_row_dict(r) for r in results if _is_failure(r)]


def _tool_skip_rows(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    return [
        _row_dict(r)
        for r in results
        if r.must_use_tool and r.tool_skip and not r.clarification_ok
    ]


def _wrong_tool_rows(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    return [_row_dict(r) for r in results if r.wrong_tool]


def _ambiguous_rows(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    return [_row_dict(r) for r in results if r.category == "ambiguous"]


def _clarification_rows(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    return [_row_dict(r) for r in results if r.clarification_ok]


def _tool_confusion_rows(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Confusion matrix on (first expected, first actual) per case.

    Only rows where the actual tool didn't match any expected tool are
    interesting for confusion. Cases with no expected tools are excluded.
    """
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    counts: Counter[tuple[str, str]] = Counter()

    for r in results:
        expected = list(r.expected_tools_json or [])
        actuals = _actual_tool_names(r)
        if not expected:
            continue
        actual_set = set(actuals)
        # Only count cases where no expected tool was hit.
        if actual_set & set(expected):
            continue
        exp_key = expected[0]
        act_key = actuals[0] if actuals else "(none)"
        key = (exp_key, act_key)
        counts[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(r.case_id)

    rows: list[dict[str, Any]] = []
    for (exp, act), n in counts.most_common():
        rows.append(
            {
                "expected_tool": exp,
                "actual_tool": act,
                "count": n,
                "example_case_ids": examples[(exp, act)],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def _serialize_summary(bundle: ReportBundle) -> dict[str, Any]:
    return {
        "run_id": bundle.run_id,
        "name": bundle.name,
        "mode": bundle.mode,
        "model": bundle.model,
        "eval_file": bundle.eval_file,
        "started_at": bundle.started_at.isoformat() if bundle.started_at else None,
        "ended_at": bundle.ended_at.isoformat() if bundle.ended_at else None,
        "total_cases": bundle.total_cases,
        "metrics": bundle.metrics,
        "latency": bundle.latency,
        "category_breakdown": bundle.category_breakdown,
        "domain_breakdown": bundle.domain_breakdown,
        "risk_breakdown": bundle.risk_breakdown,
        "failure_example_counts": {
            name: len(rows) for name, rows in bundle.failure_examples_by_signal.items()
        },
        "counts": {
            "failures": len(bundle.failures),
            "tool_skips": len(bundle.tool_skips),
            "wrong_tools": len(bundle.wrong_tools),
            "ambiguous": len(bundle.ambiguous),
            "clarifications": len(bundle.clarifications),
            "tool_confusion_pairs": len(bundle.tool_confusion),
        },
    }


def _serialize_domain_metrics(bundle: ReportBundle) -> dict[str, Any]:
    """Self-contained per-domain + per-risk + per-category JSON artifact."""
    return {
        "run_id": bundle.run_id,
        "name": bundle.name,
        "mode": bundle.mode,
        "model": bundle.model,
        "total_cases": bundle.total_cases,
        "overall_metrics": bundle.metrics,
        "by_domain": bundle.domain_breakdown,
        "by_risk": bundle.risk_breakdown,
        "by_category": bundle.category_breakdown,
        "failure_examples": bundle.failure_examples_by_signal,
    }


def write_report(bundle: ReportBundle, output_dir: Path) -> dict[str, Path]:
    """Write the four report artifacts. Returns a dict of file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{bundle.run_id}_summary.json"
    failures_path = output_dir / f"{bundle.run_id}_failures.csv"
    confusion_path = output_dir / f"{bundle.run_id}_tool_confusion.csv"
    domain_path = output_dir / f"{bundle.run_id}_domain_metrics.json"

    summary_path.write_text(
        json.dumps(_serialize_summary(bundle), indent=2, default=str),
        encoding="utf-8",
    )
    domain_path.write_text(
        json.dumps(_serialize_domain_metrics(bundle), indent=2, default=str),
        encoding="utf-8",
    )

    _write_failures_csv(failures_path, bundle.failures)
    _write_confusion_csv(confusion_path, bundle.tool_confusion)

    return {
        "summary_json": summary_path,
        "failures_csv": failures_path,
        "tool_confusion_csv": confusion_path,
        "domain_metrics_json": domain_path,
    }


def _write_failures_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "category",
        "must_use_tool",
        "expected_tools",
        "actual_tools",
        "tool_skip",
        "wrong_tool",
        "missing_evidence",
        "clarification_ok",
        "suspicious_unsupported_claim",
        "signals",
        "latency_ms",
        "trace_id",
        "message",
        "answer",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "expected_tools": ";".join(r.get("expected_tools", [])),
                    "actual_tools": ";".join(r.get("actual_tools", [])),
                    "signals": ";".join(r.get("signals", [])),
                    "message": _truncate(r.get("message"), 400),
                    "answer": _truncate(r.get("answer"), 400),
                }
            )


def _write_confusion_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["expected_tool", "actual_tool", "count", "example_case_ids"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "example_case_ids": ";".join(r.get("example_case_ids", [])),
                }
            )
