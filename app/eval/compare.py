"""Build baseline-vs-candidate comparison reports.

Phase 3A's PromptWall candidate shadow mode is intentionally behavior-neutral:
the chat path is identical to baseline, but a ``PromptWallCandidateDecision``
row is written alongside each trace. Phase 3B turns that data into a report
answering two questions:

1. *Did behavior actually stay neutral?* (Per-case actual-tool diff between
   the two runs, plus a side-by-side of all eight scorer metrics.)
2. *How good is the router's prediction, ignoring the chatbot's behavior?*
   (Confusion-matrix metrics on ``tool_required_predicted`` vs ground-truth
   ``must_use_tool``, plus a hit-rate on ``predicted_tools`` ∩
   ``expected_tools``.)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.eval.report import ReportBundle, build_report
from app.models import EvaluationResult, PromptWallCandidateDecision


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RouterMetrics:
    """How well the PromptWall analyzer predicted vs. ground truth."""

    predictions: int

    # Binary confusion matrix on tool_required_predicted vs must_use_tool.
    tool_required_true_positive: int
    tool_required_false_positive: int
    tool_required_true_negative: int
    tool_required_false_negative: int
    tool_required_precision: float
    tool_required_recall: float
    tool_required_accuracy: float

    # Did the predicted tool set intersect the case's expected_tools set?
    predicted_tool_hit_count: int
    predicted_tool_hit_denominator: int
    predicted_tool_hit_rate: float

    avg_confidence: float


@dataclass
class ComparisonBundle:
    baseline: ReportBundle
    candidate: ReportBundle
    router: RouterMetrics
    metric_deltas: dict[str, float]
    behavior_drift_count: int
    case_rows: list[dict[str, Any]] = field(default_factory=list)


# Metrics we want to compare across runs. Keep this list in sync with
# ``app.eval.scorer.aggregate_metrics``.
_COMPARE_METRICS = (
    "tool_called_when_required_rate",
    "tool_skip_rate",
    "expected_tool_hit_rate",
    "wrong_tool_rate",
    "missing_evidence_rate",
    "clarification_rate",
    "suspicious_unsupported_claim_rate",
    "average_latency_ms",
    "p95_latency_ms",
)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_comparison(
    session: Session, *, baseline_run_id: int, candidate_run_id: int
) -> ComparisonBundle:
    if baseline_run_id == candidate_run_id:
        raise ValueError("baseline_run_id and candidate_run_id must differ")

    baseline = build_report(session, baseline_run_id)
    candidate = build_report(session, candidate_run_id)

    metric_deltas = _compute_metric_deltas(baseline.metrics, candidate.metrics)

    # Index the baseline run's per-case results by case_id so we can pair them
    # with the candidate run.
    baseline_by_case = {
        r.case_id: r
        for r in session.execute(
            select(EvaluationResult).where(EvaluationResult.run_id == baseline_run_id)
        ).scalars()
    }
    candidate_results = list(
        session.execute(
            select(EvaluationResult).where(EvaluationResult.run_id == candidate_run_id)
        ).scalars()
    )

    # Per-candidate-result decision lookup (left-joined: not every case must
    # have a decision, though in practice shadow mode writes one per case).
    decisions_by_trace = {
        d.trace_id: d
        for d in session.execute(
            select(PromptWallCandidateDecision)
            .join(
                EvaluationResult,
                EvaluationResult.trace_id == PromptWallCandidateDecision.trace_id,
            )
            .where(EvaluationResult.run_id == candidate_run_id)
        ).scalars()
    }

    case_rows: list[dict[str, Any]] = []
    behavior_drift = 0

    # Router accumulators
    tp = fp = tn = fn = 0
    hit_count = 0
    hit_denominator = 0
    conf_sum = 0.0
    conf_n = 0

    for c in candidate_results:
        b = baseline_by_case.get(c.case_id)
        decision = decisions_by_trace.get(c.trace_id) if c.trace_id is not None else None

        baseline_tools = (
            [t.get("tool_name") for t in (b.actual_tools_json or [])] if b else []
        )
        candidate_tools = [t.get("tool_name") for t in (c.actual_tools_json or [])]
        behavior_diff = baseline_tools != candidate_tools
        if behavior_diff:
            behavior_drift += 1

        predicted_required: Optional[bool] = None
        predicted_tools: list[str] = []
        confidence: Optional[float] = None
        reason: Optional[str] = None
        if decision is not None:
            predicted_required = bool(decision.tool_required_predicted)
            predicted_tools = list(decision.predicted_tools or [])
            confidence = float(decision.confidence) if decision.confidence is not None else None
            reason = decision.reason
            conf_sum += confidence or 0.0
            conf_n += 1 if confidence is not None else 0

            actual = bool(c.must_use_tool)
            if predicted_required and actual:
                tp += 1
            elif predicted_required and not actual:
                fp += 1
            elif not predicted_required and not actual:
                tn += 1
            else:
                fn += 1

        expected_tools = list(c.expected_tools_json or [])
        if expected_tools and predicted_tools:
            hit_denominator += 1
            if set(predicted_tools) & set(expected_tools):
                hit_count += 1

        case_rows.append(
            {
                "case_id": c.case_id,
                "category": c.category,
                "message": c.message,
                "must_use_tool": bool(c.must_use_tool),
                "expected_tools": expected_tools,
                "baseline_actual_tools": baseline_tools,
                "candidate_actual_tools": candidate_tools,
                "behavior_drift": behavior_diff,
                "predicted_tool_required": predicted_required,
                "predicted_tools": predicted_tools,
                "predicted_confidence": confidence,
                "predicted_reason": reason,
                "predicted_required_correct": (
                    predicted_required == bool(c.must_use_tool)
                    if predicted_required is not None
                    else None
                ),
                "predicted_tool_hit": (
                    bool(set(predicted_tools) & set(expected_tools))
                    if expected_tools and predicted_tools
                    else None
                ),
            }
        )

    predictions = tp + fp + tn + fn
    router = RouterMetrics(
        predictions=predictions,
        tool_required_true_positive=tp,
        tool_required_false_positive=fp,
        tool_required_true_negative=tn,
        tool_required_false_negative=fn,
        tool_required_precision=_safe_rate(tp, tp + fp),
        tool_required_recall=_safe_rate(tp, tp + fn),
        tool_required_accuracy=_safe_rate(tp + tn, predictions),
        predicted_tool_hit_count=hit_count,
        predicted_tool_hit_denominator=hit_denominator,
        predicted_tool_hit_rate=_safe_rate(hit_count, hit_denominator),
        avg_confidence=(conf_sum / conf_n) if conf_n else 0.0,
    )

    return ComparisonBundle(
        baseline=baseline,
        candidate=candidate,
        router=router,
        metric_deltas=metric_deltas,
        behavior_drift_count=behavior_drift,
        case_rows=case_rows,
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_comparison(bundle: ComparisonBundle, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_id = bundle.baseline.run_id
    cand_id = bundle.candidate.run_id

    summary_path = output_dir / f"compare_{base_id}_vs_{cand_id}_summary.json"
    cases_path = output_dir / f"compare_{base_id}_vs_{cand_id}_cases.csv"

    summary_path.write_text(
        json.dumps(_serialize_summary(bundle), indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_cases_csv(cases_path, bundle.case_rows)

    return {"summary_json": summary_path, "cases_csv": cases_path}


def _serialize_summary(bundle: ComparisonBundle) -> dict[str, Any]:
    return {
        "baseline": {
            "run_id": bundle.baseline.run_id,
            "name": bundle.baseline.name,
            "mode": bundle.baseline.mode,
            "model": bundle.baseline.model,
            "total_cases": bundle.baseline.total_cases,
            "metrics": bundle.baseline.metrics,
            "latency": bundle.baseline.latency,
        },
        "candidate": {
            "run_id": bundle.candidate.run_id,
            "name": bundle.candidate.name,
            "mode": bundle.candidate.mode,
            "model": bundle.candidate.model,
            "total_cases": bundle.candidate.total_cases,
            "metrics": bundle.candidate.metrics,
            "latency": bundle.candidate.latency,
        },
        "metric_deltas": bundle.metric_deltas,
        "behavior_drift_count": bundle.behavior_drift_count,
        "router_metrics": {
            "predictions": bundle.router.predictions,
            "tool_required_true_positive": bundle.router.tool_required_true_positive,
            "tool_required_false_positive": bundle.router.tool_required_false_positive,
            "tool_required_true_negative": bundle.router.tool_required_true_negative,
            "tool_required_false_negative": bundle.router.tool_required_false_negative,
            "tool_required_precision": bundle.router.tool_required_precision,
            "tool_required_recall": bundle.router.tool_required_recall,
            "tool_required_accuracy": bundle.router.tool_required_accuracy,
            "predicted_tool_hit_count": bundle.router.predicted_tool_hit_count,
            "predicted_tool_hit_denominator": bundle.router.predicted_tool_hit_denominator,
            "predicted_tool_hit_rate": bundle.router.predicted_tool_hit_rate,
            "avg_confidence": bundle.router.avg_confidence,
        },
    }


def _write_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "category",
        "must_use_tool",
        "expected_tools",
        "baseline_actual_tools",
        "candidate_actual_tools",
        "behavior_drift",
        "predicted_tool_required",
        "predicted_tools",
        "predicted_confidence",
        "predicted_reason",
        "predicted_required_correct",
        "predicted_tool_hit",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "expected_tools": ";".join(r.get("expected_tools", []) or []),
                    "baseline_actual_tools": ";".join(
                        r.get("baseline_actual_tools", []) or []
                    ),
                    "candidate_actual_tools": ";".join(
                        r.get("candidate_actual_tools", []) or []
                    ),
                    "predicted_tools": ";".join(r.get("predicted_tools", []) or []),
                    "message": _truncate(r.get("message"), 400),
                }
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_rate(n: int, d: int) -> float:
    return float(n) / d if d else 0.0


def _compute_metric_deltas(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in _COMPARE_METRICS:
        b = baseline_metrics.get(key)
        c = candidate_metrics.get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            deltas[key] = float(c) - float(b)
    return deltas


def _truncate(text: Optional[str], n: int = 400) -> str:
    if not text:
        return ""
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"not serializable: {type(obj).__name__}")
