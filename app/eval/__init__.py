"""Evaluation: deterministic scorer, benchmark runner, and report builder."""

from app.eval.compare import (
    ComparisonBundle,
    RouterMetrics,
    build_comparison,
    write_comparison,
)
from app.eval.report import ReportBundle, build_report, write_report
from app.eval.runner import BenchmarkRunner, RunSummary, load_cases
from app.eval.scorer import CaseScore, ToolInvocationSummary, score_case

__all__ = [
    "BenchmarkRunner",
    "RunSummary",
    "load_cases",
    "CaseScore",
    "ToolInvocationSummary",
    "score_case",
    "ReportBundle",
    "build_report",
    "write_report",
    "ComparisonBundle",
    "RouterMetrics",
    "build_comparison",
    "write_comparison",
]
