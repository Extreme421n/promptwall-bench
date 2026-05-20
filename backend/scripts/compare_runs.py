"""Compare two benchmark runs (typically baseline vs promptwall_candidate_shadow).

The comparison includes:

* Side-by-side scorer metrics with deltas.
* A behaviour-drift count (number of cases where the chatbot called a
  different set of tools across the two runs — expected to be 0 in pure
  shadow mode, but worth measuring).
* Router-prediction accuracy on the candidate run: precision/recall/accuracy
  of ``tool_required_predicted`` against the eval ground-truth
  ``must_use_tool``, plus a hit-rate on ``predicted_tools`` vs
  ``expected_tools``.

Usage:
    python backend/scripts/compare_runs.py \\
        --baseline-run-id 1 --candidate-run-id 2 [--output-dir reports] [--top 10]
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
from app.eval import ComparisonBundle, build_comparison, write_comparison  # noqa: E402


def _hr(title: str = "") -> str:
    bar = "─" * 78
    return f"\n{bar}\n{title}\n{bar}" if title else f"\n{bar}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def _fmt_delta(value: float, *, is_pct: bool) -> str:
    sign = "+" if value >= 0 else ""
    if is_pct:
        return f"{sign}{value * 100:5.1f}pp"
    return f"{sign}{value:6.1f}"


def _print_comparison(bundle: ComparisonBundle) -> None:
    b = bundle.baseline
    c = bundle.candidate

    print(_hr("benchmark comparison"))
    print(f"baseline   #{b.run_id}  {b.name} (mode={b.mode}, model={b.model})")
    print(f"candidate  #{c.run_id}  {c.name} (mode={c.mode}, model={c.model})")
    print(f"cases:      baseline={b.total_cases}, candidate={c.total_cases}")

    print(_hr("metrics"))
    rate_keys = (
        "tool_called_when_required_rate",
        "tool_skip_rate",
        "expected_tool_hit_rate",
        "wrong_tool_rate",
        "missing_evidence_rate",
        "clarification_rate",
        "suspicious_unsupported_claim_rate",
    )
    print(f"  {'metric':<36} {'baseline':>10} {'candidate':>10}   delta")
    for k in rate_keys:
        b_v = b.metrics.get(k, 0.0)
        c_v = c.metrics.get(k, 0.0)
        d = bundle.metric_deltas.get(k, c_v - b_v)
        print(
            f"  {k:<36} {_fmt_pct(b_v):>10} {_fmt_pct(c_v):>10}   {_fmt_delta(d, is_pct=True)}"
        )
    for k in ("average_latency_ms", "p95_latency_ms"):
        b_v = b.metrics.get(k, 0.0)
        c_v = c.metrics.get(k, 0.0)
        d = bundle.metric_deltas.get(k, c_v - b_v)
        print(
            f"  {k:<36} {b_v:>10.1f} {c_v:>10.1f}   {_fmt_delta(d, is_pct=False)}"
        )

    print(_hr("behaviour drift (per-case actual tool set)"))
    drift = bundle.behavior_drift_count
    print(
        f"  cases where baseline.tools_called != candidate.tools_called: {drift}"
    )

    print(_hr("router prediction (candidate run only)"))
    r = bundle.router
    print(f"  predictions:                 {r.predictions}")
    print(f"  tool_required precision:     {_fmt_pct(r.tool_required_precision)}")
    print(f"  tool_required recall:        {_fmt_pct(r.tool_required_recall)}")
    print(f"  tool_required accuracy:      {_fmt_pct(r.tool_required_accuracy)}")
    print(
        "  confusion: "
        f"TP={r.tool_required_true_positive}, "
        f"FP={r.tool_required_false_positive}, "
        f"TN={r.tool_required_true_negative}, "
        f"FN={r.tool_required_false_negative}"
    )
    print(
        f"  predicted_tool_hit_rate:     {_fmt_pct(r.predicted_tool_hit_rate)} "
        f"({r.predicted_tool_hit_count}/{r.predicted_tool_hit_denominator})"
    )
    print(f"  avg confidence:              {r.avg_confidence:.2f}")


def _print_top_disagreements(bundle: ComparisonBundle, top: int) -> None:
    drifted = [r for r in bundle.case_rows if r["behavior_drift"]]
    print(_hr(f"top behaviour-drift cases (showing up to {top})"))
    if not drifted:
        print("  (none — chat behaviour is identical across runs)")
    else:
        for r in drifted[:top]:
            print(
                f"  [{r['case_id']}] {r['category']:<16}  "
                f"baseline={r['baseline_actual_tools']}  candidate={r['candidate_actual_tools']}"
            )

    print(_hr(f"top router misses — predicted_required != must_use_tool ({top})"))
    misses = [
        r
        for r in bundle.case_rows
        if r.get("predicted_required_correct") is False
    ]
    if not misses:
        print("  (none — every prediction matched the eval ground truth)")
    else:
        for r in misses[:top]:
            print(
                f"  [{r['case_id']}] {r['category']:<16}  "
                f"predicted={r['predicted_tool_required']}, must_use={r['must_use_tool']}"
            )
            print(f"      msg: {r['message'][:120]}")

    print(_hr(f"top tool misses — predicted_tools missed expected ({top})"))
    tool_misses = [
        r
        for r in bundle.case_rows
        if r.get("predicted_tool_hit") is False
    ]
    if not tool_misses:
        print("  (none)")
    else:
        for r in tool_misses[:top]:
            expected = ",".join(r["expected_tools"]) or "-"
            predicted = ",".join(r["predicted_tools"]) or "-"
            print(
                f"  [{r['case_id']}] {r['category']:<16}  "
                f"predicted={predicted}  expected={expected}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two benchmark runs.")
    parser.add_argument("--baseline-run-id", type=int, required=True)
    parser.add_argument("--candidate-run-id", type=int, required=True)
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL.")
    parser.add_argument("--output-dir", default="reports", help="Where to write JSON/CSV.")
    parser.add_argument("--top", type=int, default=10, help="Rows per printed section.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = make_engine(args.db_url)

    with Session(engine) as session:
        try:
            bundle = build_comparison(
                session,
                baseline_run_id=args.baseline_run_id,
                candidate_run_id=args.candidate_run_id,
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    _print_comparison(bundle)
    _print_top_disagreements(bundle, args.top)

    paths = write_comparison(bundle, Path(args.output_dir))
    print(_hr("artifacts written"))
    for kind, path in paths.items():
        print(f"  {kind:<14} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
