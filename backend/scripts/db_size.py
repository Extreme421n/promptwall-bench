"""Report on the current size of the benchmark database.

For SQLite, the size comes straight from the file on disk. For PostgreSQL,
``pg_database_size()`` is queried. Per-table row counts are reported for both
backends; per-table on-disk size is reported only for Postgres (where
``pg_total_relation_size`` is available).

Usage:
    python backend/scripts/db_size.py
    DATABASE_URL=sqlite:///./bench.db python backend/scripts/db_size.py
    python backend/scripts/db_size.py --db-url postgresql+psycopg://user:pw@host/db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Allow running as a plain script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import func, select, text  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.models import (  # noqa: E402
    Airport,
    ApiUsageDaily,
    BaggageRule,
    Base,
    Booking,
    ChatSession,
    CommerceOrder,
    CommerceOrderItem,
    CommerceRefund,
    CommerceReturn,
    Customer,
    CustomerOrganization,
    EvaluationResult,
    EvaluationRun,
    Flight,
    InternalAgentNote,
    Invoice,
    InvoiceItem,
    KBArticle,
    LLMCall,
    LoyaltyAccount,
    OperationalIncident,
    Organization,
    OverageCharge,
    Plan,
    PolicyClause,
    PolicyDocument,
    Product,
    ProductAttribute,
    ProductCategory,
    ProductInventory,
    ProductPrice,
    ProductReturnRule,
    ProductWarrantyTerms,
    PromptWallCandidateDecision,
    Refund,
    Seat,
    SeatAllocation,
    Shipment,
    Subscription,
    SupportMessage,
    SupportResolutionTemplate,
    SupportTicket,
    ToolInvocation,
    Trace,
    UsageEvent,
    Warehouse,
)

# Display order: data tables first, observability + evaluation last.
_TABLES = [
    # CRM
    Customer, LoyaltyAccount,
    # Airline
    Airport, Flight, Seat, Booking, BaggageRule, Refund,
    # Support
    SupportTicket, SupportMessage,
    # Knowledge base
    KBArticle,
    # SaaS / billing
    Organization, CustomerOrganization, Plan, Subscription,
    Invoice, InvoiceItem, UsageEvent, ApiUsageDaily,
    SeatAllocation, OverageCharge,
    # Commerce / orders
    ProductCategory, Product, ProductAttribute, ProductPrice,
    Warehouse, ProductInventory,
    CommerceOrder, CommerceOrderItem, Shipment,
    CommerceReturn, CommerceRefund,
    # Textual knowledge (Phase 6B)
    PolicyDocument, PolicyClause,
    ProductWarrantyTerms, ProductReturnRule,
    InternalAgentNote, OperationalIncident,
    SupportResolutionTemplate,
    # Observability
    ChatSession, Trace, LLMCall, ToolInvocation,
    PromptWallCandidateDecision,
    # Evaluation
    EvaluationRun, EvaluationResult,
]


# ---------------------------------------------------------------------------
# Size lookups
# ---------------------------------------------------------------------------


def _sqlite_file_path(url: str) -> Optional[Path]:
    if not url.startswith("sqlite"):
        return None
    # sqlite:///./foo.db or sqlite:////abs/path.db
    head, _, rest = url.partition(":///")
    if not rest or rest == ":memory:":
        return None
    return Path(rest)


def _format_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if n < 1024 or unit == units[-1]:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} {units[-1]}"  # pragma: no cover - unreachable


def _db_total_size_bytes(engine: Engine) -> Optional[int]:
    url = str(engine.url)
    if url.startswith("sqlite"):
        path = _sqlite_file_path(url)
        if path is None or not path.exists():
            return None
        # Include WAL/journal files that may exist alongside the DB.
        size = path.stat().st_size
        for sibling in (path.with_suffix(path.suffix + "-wal"), path.with_suffix(path.suffix + "-shm")):
            if sibling.exists():
                size += sibling.stat().st_size
        return size
    if "postgresql" in url:
        with engine.connect() as conn:
            return int(
                conn.execute(text("SELECT pg_database_size(current_database())")).scalar() or 0
            )
    return None


def _table_size_bytes(engine: Engine, table_name: str) -> Optional[int]:
    url = str(engine.url)
    if "postgresql" not in url:
        return None
    with engine.connect() as conn:
        return int(
            conn.execute(text(f"SELECT pg_total_relation_size('{table_name}')")).scalar() or 0
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report DB size and row counts.")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine = make_engine(args.db_url)
    url = str(engine.url)

    print(f"database url: {url}")

    total = _db_total_size_bytes(engine)
    if total is None:
        print("total size:   (unknown — DB file not yet created or unsupported backend)")
    else:
        print(f"total size:   {_format_bytes(total)} ({total:,} bytes)")

    print()
    is_pg = "postgresql" in url
    header = f"{'table':<22} {'rows':>14}"
    if is_pg:
        header += f"  {'on-disk':>12}"
    print(header)
    print("-" * len(header))

    with Session(engine) as session:
        grand_rows = 0
        for tbl in _TABLES:
            try:
                rows = session.execute(select(func.count()).select_from(tbl)).scalar_one()
            except Exception:  # noqa: BLE001 - table may not exist on stale DBs
                rows = 0
            grand_rows += rows
            line = f"{tbl.__tablename__:<22} {rows:>14,}"
            if is_pg:
                size = _table_size_bytes(engine, tbl.__tablename__) or 0
                line += f"  {_format_bytes(size):>12}"
            print(line)

    print("-" * len(header))
    summary = f"{'total rows':<22} {grand_rows:>14,}"
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
