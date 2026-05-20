"""Run the benchmark over an eval JSONL file.

Usage:
    python backend/scripts/run_benchmark.py \\
        --mode baseline \\
        --cases data/eval/eval_cases.jsonl \\
        --limit 100 \\
        --concurrency 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.db import make_engine  # noqa: E402
from app.eval import BenchmarkRunner, load_cases  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the chatbot benchmark.")
    parser.add_argument("--mode", default="baseline", help="Chat mode.")
    parser.add_argument("--model", default="mock", help="LLM model to use.")
    parser.add_argument(
        "--cases",
        default="data/eval/eval_cases.jsonl",
        help="Path to eval cases JSONL.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases."
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Worker threads (1 = sequential)."
    )
    parser.add_argument(
        "--db-url", default=None, help="Override DATABASE_URL for the benchmark DB."
    )
    parser.add_argument("--name", default=None, help="Run name override.")
    return parser.parse_args(argv)


def _print_summary(summary, eval_path: str) -> None:
    m = summary.metrics
    print(f"\n=== benchmark run #{summary.run_id} ({summary.name}) ===")
    print(f"eval file: {eval_path}")
    print(f"cases:     {summary.total_cases}")
    print(f"required:  {m['tool_required_cases']}")
    print()
    print(f"  tool_called_when_required_rate     {m['tool_called_when_required_rate']:.1%}")
    print(f"  tool_skip_rate                     {m['tool_skip_rate']:.1%}")
    print(f"  expected_tool_hit_rate             {m['expected_tool_hit_rate']:.1%}")
    print(f"  wrong_tool_rate                    {m['wrong_tool_rate']:.1%}")
    print(f"  missing_evidence_rate              {m['missing_evidence_rate']:.1%}")
    print(f"  clarification_rate                 {m['clarification_rate']:.1%}")
    print(f"  suspicious_unsupported_claim_rate  {m['suspicious_unsupported_claim_rate']:.1%}")
    print(f"  average_latency_ms                 {m['average_latency_ms']:.1f}")
    print(f"  p95_latency_ms                     {m['p95_latency_ms']:.1f}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"error: eval file not found: {cases_path}", file=sys.stderr)
        return 2
    cases = load_cases(cases_path)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print("error: no cases to run", file=sys.stderr)
        return 2

    engine = make_engine(args.db_url)
    runner = BenchmarkRunner(
        engine=engine,
        mode=args.mode,
        model=args.model,
        workers=args.concurrency,
    )

    print(
        f"[bench] running {len(cases)} cases (mode={args.mode}, model={args.model}, "
        f"concurrency={args.concurrency})"
    )
    t0 = time.perf_counter()
    summary = runner.run(cases, eval_file=str(cases_path), run_name=args.name)
    dt = time.perf_counter() - t0
    print(f"[bench] done in {dt:.2f}s")
    _print_summary(summary, str(cases_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
