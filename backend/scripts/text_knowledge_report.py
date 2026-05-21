"""Print a human or JSON report on the textual knowledge tables.

Usage:
    python backend/scripts/text_knowledge_report.py
    python backend/scripts/text_knowledge_report.py --json
    python backend/scripts/text_knowledge_report.py --db-url postgresql+psycopg://user@host/db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a plain script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.models import (  # noqa: E402
    InternalAgentNote,
    OperationalIncident,
    PolicyClause,
    PolicyDocument,
    ProductReturnRule,
    ProductWarrantyTerms,
    SupportResolutionTemplate,
)


_SAMPLE_EXCERPT_CHARS = 240


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(session: Session, *, top_n_policy_types: int = 20) -> dict[str, Any]:
    """Return a structured report dict with every documented section."""

    # ---- policy_documents ----
    total_policy_documents = int(
        session.execute(select(func.count()).select_from(PolicyDocument)).scalar_one()
    )
    by_domain = {
        d: int(n)
        for d, n in session.execute(
            select(PolicyDocument.domain, func.count())
            .group_by(PolicyDocument.domain)
            .order_by(func.count().desc(), PolicyDocument.domain)
        ).all()
    }
    by_policy_type_pairs = session.execute(
        select(PolicyDocument.policy_type, func.count())
        .group_by(PolicyDocument.policy_type)
        .order_by(func.count().desc(), PolicyDocument.policy_type)
    ).all()
    by_policy_type = {pt: int(n) for pt, n in by_policy_type_pairs}
    by_policy_type_top_n = [
        {"policy_type": pt, "count": int(n)}
        for pt, n in by_policy_type_pairs[:top_n_policy_types]
    ]

    avg_policy_body_length = float(
        session.execute(
            select(func.coalesce(func.avg(func.length(PolicyDocument.body)), 0.0))
        ).scalar_one()
    )

    # ---- policy_clauses ----
    total_policy_clauses = int(
        session.execute(select(func.count()).select_from(PolicyClause)).scalar_one()
    )
    avg_clause_body_length = float(
        session.execute(
            select(func.coalesce(func.avg(func.length(PolicyClause.body)), 0.0))
        ).scalar_one()
    )

    # ---- other counts ----
    total_return_rules = int(
        session.execute(select(func.count()).select_from(ProductReturnRule)).scalar_one()
    )
    total_warranty_terms = int(
        session.execute(
            select(func.count()).select_from(ProductWarrantyTerms)
        ).scalar_one()
    )
    total_internal_notes = int(
        session.execute(
            select(func.count()).select_from(InternalAgentNote)
        ).scalar_one()
    )
    total_incidents = int(
        session.execute(
            select(func.count()).select_from(OperationalIncident)
        ).scalar_one()
    )
    total_support_templates = int(
        session.execute(
            select(func.count()).select_from(SupportResolutionTemplate)
        ).scalar_one()
    )

    # ---- samples ----
    policy_samples = [
        {
            "id": p.id,
            "domain": p.domain,
            "policy_type": p.policy_type,
            "version": p.version,
            "title": p.title,
            "excerpt": _excerpt(p.body, _SAMPLE_EXCERPT_CHARS),
        }
        for p in session.execute(
            select(PolicyDocument).order_by(PolicyDocument.id).limit(5)
        ).scalars().all()
    ]
    clause_samples = [
        {
            "id": c.id,
            "policy_document_id": c.policy_document_id,
            "clause_key": c.clause_key,
            "severity": c.severity,
            "title": c.title,
            "excerpt": _excerpt(c.body, _SAMPLE_EXCERPT_CHARS),
        }
        for c in session.execute(
            select(PolicyClause).order_by(PolicyClause.id).limit(5)
        ).scalars().all()
    ]

    return {
        "policy_documents": {
            "total": total_policy_documents,
            "by_domain": by_domain,
            "by_policy_type": by_policy_type,
            "by_policy_type_top_n": by_policy_type_top_n,
            "avg_body_length": round(avg_policy_body_length, 2),
        },
        "policy_clauses": {
            "total": total_policy_clauses,
            "avg_body_length": round(avg_clause_body_length, 2),
        },
        "product_return_rules": {"total": total_return_rules},
        "product_warranty_terms": {"total": total_warranty_terms},
        "internal_agent_notes": {"total": total_internal_notes},
        "operational_incidents": {"total": total_incidents},
        "support_resolution_templates": {"total": total_support_templates},
        "samples": {
            "policy_documents": policy_samples,
            "policy_clauses": clause_samples,
        },
    }


# ---------------------------------------------------------------------------
# Helpers + printers
# ---------------------------------------------------------------------------


def _excerpt(text: str | None, max_len: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def _hr(title: str = "") -> str:
    bar = "─" * 78
    return f"\n{bar}\n{title}\n{bar}" if title else f"\n{bar}"


def _print_text_report(report: dict[str, Any]) -> None:
    pd = report["policy_documents"]
    pc = report["policy_clauses"]

    print(_hr("text knowledge report"))
    print(f"total policy documents:               {pd['total']:>6,}")
    print(f"total policy clauses:                 {pc['total']:>6,}")
    print(f"avg policy body length:               {pd['avg_body_length']:>6.0f} chars")
    print(f"avg clause body length:               {pc['avg_body_length']:>6.0f} chars")
    print(f"product return rule count:            {report['product_return_rules']['total']:>6,}")
    print(f"warranty terms count:                 {report['product_warranty_terms']['total']:>6,}")
    print(f"internal agent notes count:           {report['internal_agent_notes']['total']:>6,}")
    print(f"operational incidents count:          {report['operational_incidents']['total']:>6,}")
    print(f"support resolution templates count:   {report['support_resolution_templates']['total']:>6,}")

    print(_hr("policy documents by domain"))
    if not pd["by_domain"]:
        print("  (none)")
    for domain, n in pd["by_domain"].items():
        print(f"  {domain:<14} {n:>6,}")

    print(_hr("policy documents by policy_type"))
    if not pd["by_policy_type"]:
        print("  (none)")
    for pt, n in pd["by_policy_type"].items():
        print(f"  {pt:<28} {n:>6,}")

    print(_hr(f"top {len(pd['by_policy_type_top_n'])} policy types"))
    if not pd["by_policy_type_top_n"]:
        print("  (none)")
    for i, row in enumerate(pd["by_policy_type_top_n"], start=1):
        print(f"  {i:>2}. {row['policy_type']:<28} {row['count']:>6,}")

    print(_hr("sample policy excerpts (5)"))
    if not report["samples"]["policy_documents"]:
        print("  (none)")
    for p in report["samples"]["policy_documents"]:
        header = f"#{p['id']} · {p['domain']} · {p['policy_type']} · v{p['version']}"
        print(f"  {header}")
        print(f"      title:   {p['title']}")
        print(f"      excerpt: {p['excerpt']}")
        print()

    print(_hr("sample clause excerpts (5)"))
    if not report["samples"]["policy_clauses"]:
        print("  (none)")
    for c in report["samples"]["policy_clauses"]:
        header = f"#{c['id']} · policy {c['policy_document_id']} · {c['clause_key']} · {c['severity']}"
        print(f"  {header}")
        print(f"      title:   {c['title']}")
        print(f"      excerpt: {c['excerpt']}")
        print()


def _print_json_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on textual knowledge tables.")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL.")
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of human-readable text."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of policy types to include in the 'top' list (default 20).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine: Engine = make_engine(args.db_url)
    with Session(engine) as session:
        report = build_report(session, top_n_policy_types=args.top)
    if args.json:
        _print_json_report(report)
    else:
        _print_text_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
