"""Print a human-readable benchmark report and export JSON/CSV artifacts.

Usage:
    python backend/scripts/report_benchmark.py --run-id 1
    python backend/scripts/report_benchmark.py --run-id 1 --top 20 --output-dir reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow running as a plain script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.eval import ReportBundle, build_report, write_report  # noqa: E402


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------


def _hr(title: str = "") -> str:
    bar = "─" * 78
    return f"\n{bar}\n{title}\n{bar}" if title else f"\n{bar}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%" if isinstance(value, (int, float)) else str(value)


def _print_summary(bundle: ReportBundle) -> None:
    print(_hr(f"benchmark report — run #{bundle.run_id} ({bundle.name})"))
    print(f"mode:         {bundle.mode}")
    print(f"model:        {bundle.model}")
    print(f"eval file:    {bundle.eval_file or '-'}")
    print(f"cases:        {bundle.total_cases}")
    print(f"started:      {bundle.started_at}")
    print(f"ended:        {bundle.ended_at}")

    print(_hr("metrics"))
    m = bundle.metrics
    for key in (
        "tool_called_when_required_rate",
        "tool_skip_rate",
        "expected_tool_hit_rate",
        "wrong_tool_rate",
        "missing_evidence_rate",
        "clarification_rate",
        "suspicious_unsupported_claim_rate",
    ):
        if key in m:
            print(f"  {key:<36} {_fmt_pct(m[key])}")

    print(_hr("latency"))
    lat = bundle.latency
    print(
        f"  count={lat['count']}  "
        f"avg={lat['avg_ms']:.1f}ms  "
        f"p50={lat['p50_ms']:.1f}ms  "
        f"p95={lat['p95_ms']:.1f}ms  "
        f"p99={lat['p99_ms']:.1f}ms  "
        f"min={lat['min_ms']:.1f}ms  max={lat['max_ms']:.1f}ms"
    )

    _print_breakdown("domain breakdown", bundle.domain_breakdown, key_width=18)
    _print_breakdown("risk breakdown", bundle.risk_breakdown, key_width=18)
    _print_breakdown("category breakdown", bundle.category_breakdown, key_width=22)


def _print_breakdown(
    title: str, breakdown: dict[str, dict[str, Any]], *, key_width: int
) -> None:
    print(_hr(title))
    header = (
        f"  {'key':<{key_width}} {'n':>4} {'req':>4} "
        f"{'called%':>9} {'skip%':>8} {'hit%':>8} {'wrong%':>8} {'miss%':>8} "
        f"{'clar%':>8} {'susp%':>8} {'fail':>5}"
    )
    print(header)
    for key in sorted(breakdown):
        c = breakdown[key]
        print(
            f"  {key:<{key_width}} {c['total']:>4} {c['tool_required']:>4} "
            f"{_fmt_pct(c['tool_called_when_required_rate'])} "
            f"{_fmt_pct(c['tool_skip_rate'])} "
            f"{_fmt_pct(c['expected_tool_hit_rate'])} "
            f"{_fmt_pct(c['wrong_tool_rate'])} "
            f"{_fmt_pct(c['missing_evidence_rate'])} "
            f"{_fmt_pct(c['clarification_rate'])} "
            f"{_fmt_pct(c['suspicious_unsupported_claim_rate'])} "
            f"{c['failures']:>5}"
        )


def _print_rows(title: str, rows: list[dict[str, Any]], top: int) -> None:
    print(_hr(f"{title} ({len(rows)} total, showing up to {top})"))
    if not rows:
        print("  (none)")
        return
    for r in rows[:top]:
        signals = ",".join(r.get("signals", [])) or "-"
        expected = ",".join(r.get("expected_tools", [])) or "-"
        actual = ",".join(r.get("actual_tools", [])) or "-"
        print(
            f"  [{r['case_id']}] {r['category']:<16} "
            f"signals={signals:<48}"
        )
        print(f"      expected={expected}  actual={actual}")
        if r.get("message"):
            print(f"      msg:    {r['message'][:120]}")
        if r.get("answer"):
            print(f"      answer: {r['answer'][:120]}")


def _print_failure_buckets(bundle: ReportBundle, top: int) -> None:
    """Print the five diagnostic buckets (tool_skipped/wrong_tool/...)."""
    for bucket, rows in bundle.failure_examples_by_signal.items():
        print(_hr(f"{bucket} ({len(rows)} shown, up to {top})"))
        if not rows:
            print("  (none)")
            continue
        for r in rows[:top]:
            expected = ",".join(r.get("expected_tools", [])) or "-"
            actual = ",".join(r.get("actual_tools", [])) or "-"
            print(f"  [{r['case_id']}] {r['category']:<22}")
            print(f"      expected={expected}  actual={actual}")
            if r.get("message"):
                print(f"      msg:    {r['message'][:120]}")
            if r.get("answer"):
                print(f"      answer: {r['answer'][:120]}")


def _print_tool_confusion(bundle: ReportBundle, top: int) -> None:
    rows = bundle.tool_confusion
    print(_hr(f"tool confusion (top {top})"))
    if not rows:
        print("  (no confusions — every expected-tool case hit at least one expected tool)")
        return
    print(f"  {'expected_tool':<32} {'actual_tool':<32} {'count':>6}  examples")
    for r in rows[:top]:
        ex = ", ".join(r["example_case_ids"][:3])
        print(f"  {r['expected_tool']:<32} {r['actual_tool']:<32} {r['count']:>6}  {ex}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on a benchmark run.")
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL.")
    parser.add_argument("--output-dir", default="reports", help="Where to write JSON/CSV.")
    parser.add_argument("--top", type=int, default=10, help="Rows to print per section.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = make_engine(args.db_url)

    with Session(engine) as session:
        try:
            bundle = build_report(session, args.run_id)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    _print_summary(bundle)
    _print_rows("top failed cases", bundle.failures, args.top)
    _print_failure_buckets(bundle, args.top)
    _print_tool_confusion(bundle, args.top)

    paths = write_report(bundle, Path(args.output_dir))
    print(_hr("artifacts written"))
    for kind, path in paths.items():
        print(f"  {kind:<22} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
