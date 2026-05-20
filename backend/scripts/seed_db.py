"""CLI wrapper around ``app.seed.seed``.

Usage:
    python backend/scripts/seed_db.py --reset --small
    DATABASE_URL=sqlite:///./bench.db python backend/scripts/seed_db.py --reset --small
    python backend/scripts/seed_db.py --reset --small --db-url sqlite:///./bench.db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script (without `pip install -e .`)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.db import make_engine  # noqa: E402
from app.seed import SCALES, reset_schema, seed  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the demo benchmark DB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop and recreate every table before seeding",
    )
    scale_group = parser.add_mutually_exclusive_group()
    for name in SCALES:
        scale_group.add_argument(
            f"--{name}",
            dest="scale",
            action="store_const",
            const=name,
            help=f"use the '{name}' scale preset",
        )
    parser.add_argument(
        "--db-url",
        default=None,
        help="override DATABASE_URL (e.g. sqlite:///./bench.db)",
    )
    args = parser.parse_args(argv)
    if args.scale is None:
        args.scale = "small"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = make_engine(args.db_url)
    if args.reset:
        print(f"[seed] resetting schema on {engine.url!r}")
        reset_schema(engine)
    print(f"[seed] seeding scale={args.scale!r} on {engine.url!r}")
    t0 = time.perf_counter()
    summary = seed(engine, scale=args.scale)
    dt = time.perf_counter() - t0
    print(f"[seed] done in {dt:.2f}s")
    for k, v in summary.items():
        print(f"  {k:<22} {v:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
