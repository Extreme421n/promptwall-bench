"""Tests for backend/scripts/data_quality_report.py (Phase 6C-2)."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from backend.scripts.data_quality_report import (
    _DEFAULTS,
    build_report,
    main,
)


# Expected keys, in order, so the test breaks loudly if anyone removes one.
EXPECTED_TEXT_KEYS = {
    "policy_documents_count",
    "policy_clauses_count",
    "product_return_rules_count",
    "product_warranty_terms_count",
    "internal_agent_notes_count",
    "operational_incidents_count",
    "support_resolution_templates_count",
    "average_policy_body_length",
    "average_clause_body_length",
    "average_return_rule_body_length",
    "average_warranty_body_length",
    "empty_policy_bodies_count",
    "empty_clause_bodies_count",
    "empty_return_rule_bodies_count",
    "empty_warranty_bodies_count",
    "policies_without_clauses_count",
    "duplicate_policy_titles_by_domain_version_count",
}

EXPECTED_RELATIONSHIP_KEYS = {
    "orphan_policy_clauses_count",
    "orphan_internal_notes_count",
    "products_without_warranty_terms_count",
    "product_categories_without_return_rules_count",
    "support_tickets_without_customer_count",
}

EXPECTED_OPERATIONAL_KEYS = {
    "stale_flight_status_count",
    "pending_refunds_past_due_count",
    "invoice_status_mismatch_count",
    "closed_ticket_open_refund_count",
    "missing_tracking_number_count",
    "duplicate_customer_email_count",
}


# ---------------------------------------------------------------------------
# build_report — invoked directly with the small-seed session
# ---------------------------------------------------------------------------


@pytest.fixture()
def report(seeded_session: Session) -> dict[str, Any]:
    return build_report(seeded_session)


def test_report_top_level_keys(report: dict[str, Any]) -> None:
    assert {
        "generated_at",
        "thresholds",
        "text_knowledge",
        "relationships",
        "operational",
        "totals",
        "warnings",
    } <= set(report)


def test_text_knowledge_keys(report: dict[str, Any]) -> None:
    missing = EXPECTED_TEXT_KEYS - set(report["text_knowledge"])
    assert not missing, f"missing text-knowledge keys: {missing}"


def test_relationship_keys(report: dict[str, Any]) -> None:
    missing = EXPECTED_RELATIONSHIP_KEYS - set(report["relationships"])
    assert not missing, f"missing relationship keys: {missing}"


def test_operational_keys(report: dict[str, Any]) -> None:
    missing = EXPECTED_OPERATIONAL_KEYS - set(report["operational"])
    assert not missing, f"missing operational keys: {missing}"


def test_thresholds_carry_defaults(report: dict[str, Any]) -> None:
    for k in _DEFAULTS:
        assert k in report["thresholds"]


def test_warnings_is_a_list(report: dict[str, Any]) -> None:
    assert isinstance(report["warnings"], list)
    for w in report["warnings"]:
        assert {"severity", "metric", "value", "threshold", "message"} <= set(w)
        assert w["severity"] in ("warning", "error")


# ---------------------------------------------------------------------------
# Small-seed contract: counts match the seed's known volumes
# ---------------------------------------------------------------------------


def test_small_seed_counts_are_populated(report: dict[str, Any]) -> None:
    tk = report["text_knowledge"]
    assert tk["policy_documents_count"] == 50
    assert tk["policy_clauses_count"] == 300
    assert tk["product_return_rules_count"] == 100
    assert tk["product_warranty_terms_count"] == 100
    assert tk["internal_agent_notes_count"] == 500
    assert tk["operational_incidents_count"] == 50
    assert tk["support_resolution_templates_count"] == 100


def test_no_empty_bodies_on_small_seed(report: dict[str, Any]) -> None:
    tk = report["text_knowledge"]
    assert tk["empty_policy_bodies_count"] == 0
    assert tk["empty_clause_bodies_count"] == 0
    assert tk["empty_return_rule_bodies_count"] == 0
    assert tk["empty_warranty_bodies_count"] == 0


def test_no_orphans_on_small_seed(report: dict[str, Any]) -> None:
    rel = report["relationships"]
    assert rel["orphan_policy_clauses_count"] == 0
    assert rel["orphan_internal_notes_count"] == 0
    assert rel["products_without_warranty_terms_count"] == 0
    assert rel["product_categories_without_return_rules_count"] == 0
    assert rel["support_tickets_without_customer_count"] == 0


def test_avg_lengths_are_floats(report: dict[str, Any]) -> None:
    tk = report["text_knowledge"]
    for key in (
        "average_policy_body_length",
        "average_clause_body_length",
        "average_return_rule_body_length",
        "average_warranty_body_length",
    ):
        v = tk[key]
        assert isinstance(v, float), f"{key} should be a float, got {type(v)}"
        assert v > 0, f"{key} should be > 0, got {v}"


# ---------------------------------------------------------------------------
# Unsupported checks — must not crash, must report 'not_available'
# ---------------------------------------------------------------------------


def test_unsupported_check_falls_back_to_not_available(
    report: dict[str, Any],
) -> None:
    """``closed_ticket_open_refund_count`` requires a column on SupportTicket
    that isn't in the current schema. The report must surface this gracefully
    as ``"not_available"`` rather than blowing up."""
    op = report["operational"]
    assert op["closed_ticket_open_refund_count"] == "not_available"


def test_no_check_raises_on_normal_seed(report: dict[str, Any]) -> None:
    """Every check that ran must either be a number, a string, or
    not_available. No exceptions/None leaks."""
    for section_key in ("text_knowledge", "relationships", "operational"):
        for key, val in report[section_key].items():
            assert isinstance(val, (int, float, str)), (
                f"{section_key}.{key} has unexpected type {type(val).__name__}: {val!r}"
            )


# ---------------------------------------------------------------------------
# Threshold behaviour — warnings actually fire when limits are breached
# ---------------------------------------------------------------------------


def test_warning_fires_when_avg_policy_body_threshold_raised(
    seeded_session: Session,
) -> None:
    """Bump the threshold so the small-seed average (~150 chars) is below it,
    and verify a warning fires."""
    r = build_report(
        seeded_session, thresholds={"min_avg_policy_body_length": 10_000}
    )
    fired = [w for w in r["warnings"] if w["metric"] == "average_policy_body_length"]
    assert fired, "expected a warning when threshold is raised above seed average"
    assert fired[0]["severity"] == "warning"


def test_warning_does_not_fire_when_threshold_is_lenient(
    seeded_session: Session,
) -> None:
    r = build_report(
        seeded_session,
        thresholds={
            "min_avg_policy_body_length": 1,
            "min_avg_clause_body_length": 1,
            "max_empty_clauses_ratio": 1.0,
            "max_products_without_warranty_ratio": 1.0,
            "max_duplicate_emails": 10_000,
        },
    )
    assert r["warnings"] == []


# ---------------------------------------------------------------------------
# CLI smoke — text and JSON
# ---------------------------------------------------------------------------


def test_cli_text_mode_runs_on_small_seed(seeded_engine, capsys) -> None:
    engine, _ = seeded_engine
    rc = main(["--db-url", str(engine.url)])
    out = capsys.readouterr().out
    assert "DemoCorp data quality report" in out
    assert "1. text knowledge" in out
    assert "warnings" in out.lower()
    assert rc in (0, 1)


def test_cli_json_mode_emits_valid_json(seeded_engine, capsys) -> None:
    engine, _ = seeded_engine
    rc = main(["--db-url", str(engine.url), "--json"])
    out = capsys.readouterr().out
    data = json.loads(out)  # must parse cleanly
    assert "warnings" in data
    assert "text_knowledge" in data
    assert "relationships" in data
    assert "operational" in data
    # Match-metadata: every text_knowledge key from the spec is present.
    assert EXPECTED_TEXT_KEYS <= set(data["text_knowledge"])
    assert EXPECTED_RELATIONSHIP_KEYS <= set(data["relationships"])
    assert EXPECTED_OPERATIONAL_KEYS <= set(data["operational"])
    assert rc in (0, 1)


def test_cli_returns_non_zero_on_error_severity(
    monkeypatch, seeded_engine, capsys
) -> None:
    """If we inject an orphan-style 'error' warning, main() must exit non-zero
    so CI users can rely on the exit code."""
    from backend.scripts import data_quality_report as dq

    real_build = dq.build_report

    def _injecting_build(session, thresholds=None):  # noqa: ANN001
        r = real_build(session, thresholds=thresholds)
        r["warnings"].append(
            {
                "severity": "error",
                "metric": "orphan_test",
                "value": 1,
                "threshold": 0,
                "message": "injected for test",
            }
        )
        return r

    monkeypatch.setattr(dq, "build_report", _injecting_build)
    engine, _ = seeded_engine
    rc = dq.main(["--db-url", str(engine.url), "--json"])
    assert rc == 1
