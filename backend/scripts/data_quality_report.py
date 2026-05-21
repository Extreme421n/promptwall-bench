"""Data-quality report for the DemoCorp simulation database.

Usage:
    python -m backend.scripts.data_quality_report
    python -m backend.scripts.data_quality_report --json
    python -m backend.scripts.data_quality_report --db-url postgresql+psycopg://user@host/db

The report verifies that the demo DB is rich, realistic, and structurally
consistent. It covers three areas:

  1. Text-knowledge volume + text-length stats + empty-body detection.
  2. Relationship integrity (orphans, products without warranties, categories
     without return rules, …).
  3. Operational data consistency (past-due refunds, stale flight status,
     missing tracking numbers, duplicate customer emails, …).

Each metric is computed defensively: if the underlying table or column is not
present in the current schema, the check returns the literal string
``"not_available"`` instead of crashing. Threshold warnings are collected in
``warnings`` and emitted both in text and JSON output.

This script is read-only. It does not write to the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Allow running as a plain script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.engine import Engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import make_engine  # noqa: E402

# Models are imported defensively — anything that may be absent (older
# migrations, partial seeds) is wrapped so the report never crashes.
try:  # textual knowledge (Phase 6B-1)
    from app.models import (  # noqa: E402
        InternalAgentNote,
        OperationalIncident,
        PolicyClause,
        PolicyDocument,
        ProductCategory,
        ProductReturnRule,
        ProductWarrantyTerms,
        SupportResolutionTemplate,
    )
except Exception:  # pragma: no cover — defensive
    InternalAgentNote = None  # type: ignore[assignment]
    OperationalIncident = None  # type: ignore[assignment]
    PolicyClause = None  # type: ignore[assignment]
    PolicyDocument = None  # type: ignore[assignment]
    ProductCategory = None  # type: ignore[assignment]
    ProductReturnRule = None  # type: ignore[assignment]
    ProductWarrantyTerms = None  # type: ignore[assignment]
    SupportResolutionTemplate = None  # type: ignore[assignment]

try:
    from app.models import (  # noqa: E402
        Booking,
        Customer,
        Flight,
        Invoice,
        Product,
        Refund,
        Shipment,
        SupportTicket,
    )
except Exception:  # pragma: no cover
    Booking = None  # type: ignore[assignment]
    Customer = None  # type: ignore[assignment]
    Flight = None  # type: ignore[assignment]
    Invoice = None  # type: ignore[assignment]
    Product = None  # type: ignore[assignment]
    Refund = None  # type: ignore[assignment]
    Shipment = None  # type: ignore[assignment]
    SupportTicket = None  # type: ignore[assignment]

try:
    from app.models import CommerceRefund  # noqa: E402
except Exception:  # pragma: no cover
    CommerceRefund = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Threshold defaults — tuned to the small-seed reality so production sizes
# don't false-alarm. Each is overridable via a CLI flag.
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "min_avg_policy_body_length": 200,
    "min_avg_clause_body_length": 60,
    "max_empty_clauses_ratio": 0.05,
    "max_products_without_warranty_ratio": 0.10,
    "max_duplicate_emails": 0,  # any duplicate is suspicious
}


# ---------------------------------------------------------------------------
# Safe-call helpers
# ---------------------------------------------------------------------------


def _safe(fn: Callable[[], Any], default: Any = "not_available") -> Any:
    """Run ``fn`` and swallow any exception, returning ``default`` instead.

    Used so a missing table / column / unsupported SQL feature can never
    crash the report. We intentionally do NOT silence everything — a SQL
    *syntax* bug in this script would also become 'not_available', so we
    log a one-line diagnostic to stderr to keep development sane.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — by-design swallow
        print(
            f"  [data_quality] check failed → not_available: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return default


def _count(session: Session, model: Any) -> Any:
    if model is None:
        return "not_available"
    return _safe(
        lambda: int(session.execute(select(func.count()).select_from(model)).scalar_one())
    )


def _avg_length(session: Session, column: Any) -> Any:
    if column is None:
        return "not_available"
    return _safe(
        lambda: float(
            session.execute(
                select(func.coalesce(func.avg(func.length(column)), 0.0))
            ).scalar_one()
        )
    )


def _empty_bodies(session: Session, model: Any, column: Any) -> Any:
    """Count rows where the given column is NULL, empty, or whitespace-only."""
    if model is None or column is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count())
                .select_from(model)
                .where((column.is_(None)) | (func.length(func.trim(column)) == 0))
            ).scalar_one()
        )
    )


# ---------------------------------------------------------------------------
# Section: text-knowledge counts + body stats
# ---------------------------------------------------------------------------


def _text_knowledge_section(session: Session) -> dict[str, Any]:
    return {
        # counts (1–7)
        "policy_documents_count": _count(session, PolicyDocument),
        "policy_clauses_count": _count(session, PolicyClause),
        "product_return_rules_count": _count(session, ProductReturnRule),
        "product_warranty_terms_count": _count(session, ProductWarrantyTerms),
        "internal_agent_notes_count": _count(session, InternalAgentNote),
        "operational_incidents_count": _count(session, OperationalIncident),
        "support_resolution_templates_count": _count(
            session, SupportResolutionTemplate
        ),
        # average body lengths (8–11)
        "average_policy_body_length": _avg_length(
            session, getattr(PolicyDocument, "body", None)
        ),
        "average_clause_body_length": _avg_length(
            session, getattr(PolicyClause, "body", None)
        ),
        "average_return_rule_body_length": _avg_length(
            session, getattr(ProductReturnRule, "body", None)
        ),
        "average_warranty_body_length": _avg_length(
            session, getattr(ProductWarrantyTerms, "body", None)
        ),
        # empty-body detection (12–15)
        "empty_policy_bodies_count": _empty_bodies(
            session, PolicyDocument, getattr(PolicyDocument, "body", None)
        ),
        "empty_clause_bodies_count": _empty_bodies(
            session, PolicyClause, getattr(PolicyClause, "body", None)
        ),
        "empty_return_rule_bodies_count": _empty_bodies(
            session, ProductReturnRule, getattr(ProductReturnRule, "body", None)
        ),
        "empty_warranty_bodies_count": _empty_bodies(
            session, ProductWarrantyTerms, getattr(ProductWarrantyTerms, "body", None)
        ),
        # 16. policies without any clauses
        "policies_without_clauses_count": _policies_without_clauses(session),
        # 17. duplicate policy titles within (domain, version)
        "duplicate_policy_titles_by_domain_version_count": (
            _duplicate_policy_titles(session)
        ),
    }


def _policies_without_clauses(session: Session) -> Any:
    if PolicyDocument is None or PolicyClause is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count(PolicyDocument.id)).where(
                    ~PolicyDocument.id.in_(
                        select(PolicyClause.policy_document_id).distinct()
                    )
                )
            ).scalar_one()
        )
    )


def _duplicate_policy_titles(session: Session) -> Any:
    if PolicyDocument is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count()).select_from(
                    select(
                        PolicyDocument.domain,
                        PolicyDocument.version,
                        PolicyDocument.title,
                        func.count().label("n"),
                    )
                    .group_by(
                        PolicyDocument.domain,
                        PolicyDocument.version,
                        PolicyDocument.title,
                    )
                    .having(func.count() > 1)
                    .subquery()
                )
            ).scalar_one()
        )
    )


# ---------------------------------------------------------------------------
# Section: relationship integrity
# ---------------------------------------------------------------------------


def _relationship_section(session: Session) -> dict[str, Any]:
    return {
        # 18. clauses whose parent policy_document_id doesn't exist
        "orphan_policy_clauses_count": _orphan_policy_clauses(session),
        # 19. notes whose customer_id doesn't exist
        "orphan_internal_notes_count": _orphan_internal_notes(session),
        # 20. products with no warranty row
        "products_without_warranty_terms_count": _products_without_warranty(session),
        # 21. categories with no return rule
        "product_categories_without_return_rules_count": (
            _categories_without_return_rules(session)
        ),
        # 22. support tickets with non-existent customer_id (optional)
        "support_tickets_without_customer_count": _orphan_support_tickets(session),
    }


def _orphan_policy_clauses(session: Session) -> Any:
    if PolicyDocument is None or PolicyClause is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count(PolicyClause.id)).where(
                    ~PolicyClause.policy_document_id.in_(select(PolicyDocument.id))
                )
            ).scalar_one()
        )
    )


def _orphan_internal_notes(session: Session) -> Any:
    if Customer is None or InternalAgentNote is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count(InternalAgentNote.id)).where(
                    ~InternalAgentNote.customer_id.in_(select(Customer.id))
                )
            ).scalar_one()
        )
    )


def _products_without_warranty(session: Session) -> Any:
    if Product is None or ProductWarrantyTerms is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count(Product.id)).where(
                    ~Product.id.in_(select(ProductWarrantyTerms.product_id).distinct())
                )
            ).scalar_one()
        )
    )


def _categories_without_return_rules(session: Session) -> Any:
    if ProductCategory is None or ProductReturnRule is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count(ProductCategory.id)).where(
                    ~ProductCategory.id.in_(
                        select(ProductReturnRule.product_category_id).distinct()
                    )
                )
            ).scalar_one()
        )
    )


def _orphan_support_tickets(session: Session) -> Any:
    if SupportTicket is None or Customer is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count(SupportTicket.id)).where(
                    ~SupportTicket.customer_id.in_(select(Customer.id))
                )
            ).scalar_one()
        )
    )


# ---------------------------------------------------------------------------
# Section: operational / data consistency
# ---------------------------------------------------------------------------


def _operational_section(session: Session) -> dict[str, Any]:
    return {
        # 23. flights with scheduled_departure in the past but status=scheduled
        "stale_flight_status_count": _stale_flights(session),
        # 24. refunds in 'pending' state past expected_resolution_date
        "pending_refunds_past_due_count": _pending_refunds_past_due(session),
        # 25. invoices marked 'paid' without a paid_at timestamp (or vice-versa)
        "invoice_status_mismatch_count": _invoice_status_mismatches(session),
        # 26. closed support tickets that still have a pending refund attached
        "closed_ticket_open_refund_count": _closed_ticket_open_refund(session),
        # 27. shipments missing tracking_number (NULL or empty string)
        "missing_tracking_number_count": _missing_tracking_numbers(session),
        # 28. customer rows sharing an email address
        "duplicate_customer_email_count": _duplicate_customer_emails(session),
    }


def _stale_flights(session: Session) -> Any:
    if Flight is None:
        return "not_available"

    def _check() -> int:
        now = datetime.now(timezone.utc)
        return int(
            session.execute(
                select(func.count(Flight.id)).where(
                    Flight.scheduled_departure < now,
                    Flight.status == "scheduled",
                )
            ).scalar_one()
        )

    return _safe(_check)


def _pending_refunds_past_due(session: Session) -> Any:
    if Refund is None:
        return "not_available"

    def _check() -> int:
        today = datetime.now(timezone.utc).date()
        return int(
            session.execute(
                select(func.count(Refund.id)).where(
                    Refund.refund_status == "pending",
                    Refund.expected_resolution_date.is_not(None),
                    Refund.expected_resolution_date < today,
                )
            ).scalar_one()
        )

    return _safe(_check)


def _invoice_status_mismatches(session: Session) -> Any:
    if Invoice is None:
        return "not_available"

    def _check() -> int:
        # 'paid' invoices without paid_at OR not-'paid' invoices that have paid_at.
        paid_no_ts = int(
            session.execute(
                select(func.count(Invoice.id)).where(
                    Invoice.status == "paid", Invoice.paid_at.is_(None)
                )
            ).scalar_one()
        )
        ts_not_paid = int(
            session.execute(
                select(func.count(Invoice.id)).where(
                    Invoice.status != "paid", Invoice.paid_at.is_not(None)
                )
            ).scalar_one()
        )
        return paid_no_ts + ts_not_paid

    return _safe(_check)


def _closed_ticket_open_refund(session: Session) -> Any:
    """Joins support_tickets ↔ refunds via the shared customer_id +
    booking_reference. The exact join shape is schema-dependent; if no such
    relationship exists, returns ``not_available``."""
    if SupportTicket is None or Refund is None:
        return "not_available"

    def _check() -> int:
        # SupportTicket.related_booking_reference (if present) joins to
        # Refund.booking_reference. We probe defensively.
        related = getattr(SupportTicket, "related_booking_reference", None)
        if related is None:
            return -1  # signals "schema doesn't support this check"
        return int(
            session.execute(
                select(func.count(SupportTicket.id))
                .join(
                    Refund,
                    related == Refund.booking_reference,
                )
                .where(
                    SupportTicket.status == "closed",
                    Refund.refund_status == "pending",
                )
            ).scalar_one()
        )

    val = _safe(_check, default=-1)
    return "not_available" if val == -1 else val


def _missing_tracking_numbers(session: Session) -> Any:
    if Shipment is None:
        return "not_available"
    column = getattr(Shipment, "tracking_number", None)
    if column is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count(Shipment.id)).where(
                    (column.is_(None)) | (func.length(func.trim(column)) == 0)
                )
            ).scalar_one()
        )
    )


def _duplicate_customer_emails(session: Session) -> Any:
    if Customer is None:
        return "not_available"
    return _safe(
        lambda: int(
            session.execute(
                select(func.count()).select_from(
                    select(Customer.email, func.count().label("n"))
                    .where(Customer.email.is_not(None))
                    .group_by(Customer.email)
                    .having(func.count() > 1)
                    .subquery()
                )
            ).scalar_one()
        )
    )


# ---------------------------------------------------------------------------
# Threshold warnings
# ---------------------------------------------------------------------------


def _build_warnings(
    report: dict[str, Any], thresholds: dict[str, Any]
) -> list[dict[str, Any]]:
    """Emit a structured warnings list given the report values + thresholds.

    Each warning is a dict so JSON consumers can filter by severity / metric
    without parsing prose. Numbers that came back ``"not_available"`` are
    silently skipped — they're not warnings, they're just unsupported on this
    schema.
    """
    out: list[dict[str, Any]] = []
    tk = report["text_knowledge"]
    rel = report["relationships"]
    op = report["operational"]

    def _num(v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        return None

    # Body-length warnings.
    avg_policy = _num(tk["average_policy_body_length"])
    if avg_policy is not None and avg_policy < thresholds["min_avg_policy_body_length"]:
        out.append(
            {
                "severity": "warning",
                "metric": "average_policy_body_length",
                "value": avg_policy,
                "threshold": thresholds["min_avg_policy_body_length"],
                "message": (
                    f"average policy body length {avg_policy:.0f} chars is below "
                    f"the {thresholds['min_avg_policy_body_length']} char floor"
                ),
            }
        )
    avg_clause = _num(tk["average_clause_body_length"])
    if avg_clause is not None and avg_clause < thresholds["min_avg_clause_body_length"]:
        out.append(
            {
                "severity": "warning",
                "metric": "average_clause_body_length",
                "value": avg_clause,
                "threshold": thresholds["min_avg_clause_body_length"],
                "message": (
                    f"average clause body length {avg_clause:.0f} chars is below "
                    f"the {thresholds['min_avg_clause_body_length']} char floor"
                ),
            }
        )

    # Empty-clauses ratio.
    empty_clauses = _num(tk["empty_clause_bodies_count"])
    total_clauses = _num(tk["policy_clauses_count"])
    if empty_clauses is not None and total_clauses is not None and total_clauses > 0:
        ratio = empty_clauses / total_clauses
        if ratio > thresholds["max_empty_clauses_ratio"]:
            out.append(
                {
                    "severity": "warning",
                    "metric": "empty_clause_bodies_ratio",
                    "value": ratio,
                    "threshold": thresholds["max_empty_clauses_ratio"],
                    "message": (
                        f"{ratio:.1%} of policy clauses have an empty body "
                        f"(threshold {thresholds['max_empty_clauses_ratio']:.0%})"
                    ),
                }
            )

    # Products without warranty.
    pw_missing = _num(rel["products_without_warranty_terms_count"])
    total_products = _safe_count(Product) if Product is not None else None
    if pw_missing is not None and total_products and total_products > 0:
        ratio = pw_missing / total_products
        if ratio > thresholds["max_products_without_warranty_ratio"]:
            out.append(
                {
                    "severity": "warning",
                    "metric": "products_without_warranty_ratio",
                    "value": ratio,
                    "threshold": thresholds["max_products_without_warranty_ratio"],
                    "message": (
                        f"{ratio:.1%} of products lack a warranty row "
                        f"(threshold {thresholds['max_products_without_warranty_ratio']:.0%})"
                    ),
                }
            )

    # Orphans of any kind = error severity.
    for key in (
        "orphan_policy_clauses_count",
        "orphan_internal_notes_count",
        "support_tickets_without_customer_count",
    ):
        v = _num(rel[key])
        if v is not None and v > 0:
            out.append(
                {
                    "severity": "error",
                    "metric": key,
                    "value": int(v),
                    "threshold": 0,
                    "message": f"{int(v)} orphan record(s) detected in {key}",
                }
            )

    # Duplicate emails.
    dup = _num(op["duplicate_customer_email_count"])
    if dup is not None and dup > thresholds["max_duplicate_emails"]:
        out.append(
            {
                "severity": "warning",
                "metric": "duplicate_customer_email_count",
                "value": int(dup),
                "threshold": thresholds["max_duplicate_emails"],
                "message": (
                    f"{int(dup)} email addresses are shared by more than one customer"
                ),
            }
        )

    return out


def _safe_count(model: Any) -> Optional[int]:
    """Tiny helper for warning thresholds that need a denominator outside the
    main per-session loop. Returns ``None`` if the model is missing or any
    error occurs."""
    if model is None:
        return None
    # We don't have a session here — this helper is intentionally weak; the
    # main report already has the count, but the warning builder reaches for
    # it from a cached value if present. In practice the builder reads the
    # cached count via the report dict; this function is just defensive.
    return None


# ---------------------------------------------------------------------------
# Top-level report build
# ---------------------------------------------------------------------------


def build_report(
    session: Session, *, thresholds: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Return the full report dict (sections + warnings)."""
    thresholds = {**_DEFAULTS, **(thresholds or {})}

    text_knowledge = _text_knowledge_section(session)
    relationships = _relationship_section(session)
    operational = _operational_section(session)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": thresholds,
        "text_knowledge": text_knowledge,
        "relationships": relationships,
        "operational": operational,
    }

    # Compute the products-without-warranty ratio inline (the builder needs a
    # denominator that's already in the relationships section).
    total_products = _count(session, Product) if Product is not None else None
    report["totals"] = {
        "products": total_products,
    }

    # Patch the warning builder so it can reach the cached product count.
    warnings = _build_warnings_with_totals(report, thresholds)
    report["warnings"] = warnings
    return report


def _build_warnings_with_totals(
    report: dict[str, Any], thresholds: dict[str, Any]
) -> list[dict[str, Any]]:
    """Same as ``_build_warnings`` but pulls denominators (e.g. total products)
    from the already-computed ``report["totals"]`` instead of hitting the DB
    a second time."""
    out: list[dict[str, Any]] = []
    tk = report["text_knowledge"]
    rel = report["relationships"]
    op = report["operational"]
    totals = report.get("totals", {})

    def _num(v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        return None

    # --- body-length warnings ---
    avg_policy = _num(tk["average_policy_body_length"])
    if avg_policy is not None and avg_policy < thresholds["min_avg_policy_body_length"]:
        out.append(
            {
                "severity": "warning",
                "metric": "average_policy_body_length",
                "value": avg_policy,
                "threshold": thresholds["min_avg_policy_body_length"],
                "message": (
                    f"average policy body length {avg_policy:.0f} chars is below "
                    f"the {thresholds['min_avg_policy_body_length']} char floor"
                ),
            }
        )
    avg_clause = _num(tk["average_clause_body_length"])
    if avg_clause is not None and avg_clause < thresholds["min_avg_clause_body_length"]:
        out.append(
            {
                "severity": "warning",
                "metric": "average_clause_body_length",
                "value": avg_clause,
                "threshold": thresholds["min_avg_clause_body_length"],
                "message": (
                    f"average clause body length {avg_clause:.0f} chars is below "
                    f"the {thresholds['min_avg_clause_body_length']} char floor"
                ),
            }
        )

    # --- empty-clauses ratio ---
    empty_clauses = _num(tk["empty_clause_bodies_count"])
    total_clauses = _num(tk["policy_clauses_count"])
    if empty_clauses is not None and total_clauses is not None and total_clauses > 0:
        ratio = empty_clauses / total_clauses
        if ratio > thresholds["max_empty_clauses_ratio"]:
            out.append(
                {
                    "severity": "warning",
                    "metric": "empty_clause_bodies_ratio",
                    "value": ratio,
                    "threshold": thresholds["max_empty_clauses_ratio"],
                    "message": (
                        f"{ratio:.1%} of policy clauses have an empty body "
                        f"(threshold {thresholds['max_empty_clauses_ratio']:.0%})"
                    ),
                }
            )

    # --- products without warranty ratio ---
    pw_missing = _num(rel["products_without_warranty_terms_count"])
    total_products = _num(totals.get("products"))
    if (
        pw_missing is not None
        and total_products is not None
        and total_products > 0
    ):
        ratio = pw_missing / total_products
        if ratio > thresholds["max_products_without_warranty_ratio"]:
            out.append(
                {
                    "severity": "warning",
                    "metric": "products_without_warranty_ratio",
                    "value": ratio,
                    "threshold": thresholds["max_products_without_warranty_ratio"],
                    "message": (
                        f"{ratio:.1%} of products lack a warranty row "
                        f"(threshold {thresholds['max_products_without_warranty_ratio']:.0%})"
                    ),
                }
            )

    # --- orphans → error severity ---
    for key in (
        "orphan_policy_clauses_count",
        "orphan_internal_notes_count",
        "support_tickets_without_customer_count",
    ):
        v = _num(rel[key])
        if v is not None and v > 0:
            out.append(
                {
                    "severity": "error",
                    "metric": key,
                    "value": int(v),
                    "threshold": 0,
                    "message": f"{int(v)} orphan record(s) detected in {key}",
                }
            )

    # --- duplicate emails ---
    dup = _num(op["duplicate_customer_email_count"])
    if dup is not None and dup > thresholds["max_duplicate_emails"]:
        out.append(
            {
                "severity": "warning",
                "metric": "duplicate_customer_email_count",
                "value": int(dup),
                "threshold": thresholds["max_duplicate_emails"],
                "message": (
                    f"{int(dup)} email addresses are shared by more than one customer"
                ),
            }
        )

    return out


# ---------------------------------------------------------------------------
# Text + JSON renderers
# ---------------------------------------------------------------------------


def _hr(label: str) -> str:
    return "\n" + "─" * 78 + f"\n{label}\n" + "─" * 78


def _fmt(value: Any) -> str:
    if value == "not_available":
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _print_text(report: dict[str, Any]) -> None:
    print(_hr("DemoCorp data quality report"))
    print(f"generated_at: {report['generated_at']}")

    print(_hr("1. text knowledge"))
    for k, v in report["text_knowledge"].items():
        print(f"  {k:<48} {_fmt(v):>14}")

    print(_hr("2. relationship integrity"))
    for k, v in report["relationships"].items():
        print(f"  {k:<48} {_fmt(v):>14}")

    print(_hr("3. operational consistency"))
    for k, v in report["operational"].items():
        print(f"  {k:<48} {_fmt(v):>14}")

    print(_hr("warnings"))
    if not report["warnings"]:
        print("  no warnings — data quality looks healthy ✅")
    else:
        for w in report["warnings"]:
            sev = w["severity"].upper()
            print(f"  [{sev}] {w['metric']}: {w['message']}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DemoCorp data-quality report."
    )
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--min-policy-body",
        type=int,
        default=_DEFAULTS["min_avg_policy_body_length"],
        help="Minimum acceptable average policy body length in chars.",
    )
    parser.add_argument(
        "--min-clause-body",
        type=int,
        default=_DEFAULTS["min_avg_clause_body_length"],
        help="Minimum acceptable average clause body length in chars.",
    )
    parser.add_argument(
        "--max-empty-clauses",
        type=float,
        default=_DEFAULTS["max_empty_clauses_ratio"],
        help="Maximum tolerated empty-clauses ratio (e.g. 0.05 = 5%%).",
    )
    parser.add_argument(
        "--max-products-without-warranty",
        type=float,
        default=_DEFAULTS["max_products_without_warranty_ratio"],
        help="Maximum tolerated ratio of products without a warranty row.",
    )
    parser.add_argument(
        "--max-duplicate-emails",
        type=int,
        default=_DEFAULTS["max_duplicate_emails"],
        help="Maximum tolerated duplicate-email count.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engine: Engine = make_engine(args.db_url)
    thresholds = {
        "min_avg_policy_body_length": args.min_policy_body,
        "min_avg_clause_body_length": args.min_clause_body,
        "max_empty_clauses_ratio": args.max_empty_clauses,
        "max_products_without_warranty_ratio": args.max_products_without_warranty,
        "max_duplicate_emails": args.max_duplicate_emails,
    }
    with Session(engine) as session:
        report = build_report(session, thresholds=thresholds)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        _print_text(report)

    # Non-zero exit only if we hit an *error* severity (e.g. orphans). Warning-
    # only runs return 0 so CI can run this informationally without failing.
    has_errors = any(w["severity"] == "error" for w in report["warnings"])
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
