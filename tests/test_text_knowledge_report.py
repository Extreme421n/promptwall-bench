"""Tests for the text knowledge report CLI (Phase 6B-3)."""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# build_report (pure)
# ---------------------------------------------------------------------------


def test_build_report_returns_documented_keys(seeded_session) -> None:
    from backend.scripts.text_knowledge_report import build_report

    report = build_report(seeded_session, top_n_policy_types=20)

    # Top-level keys
    for key in (
        "policy_documents",
        "policy_clauses",
        "product_return_rules",
        "product_warranty_terms",
        "internal_agent_notes",
        "operational_incidents",
        "support_resolution_templates",
        "samples",
    ):
        assert key in report, f"missing top-level key: {key!r}"

    # Nested keys for policy_documents
    pd = report["policy_documents"]
    for key in ("total", "by_domain", "by_policy_type", "by_policy_type_top_n", "avg_body_length"):
        assert key in pd, f"missing policy_documents.{key!r}"

    # Nested keys for policy_clauses
    pc = report["policy_clauses"]
    for key in ("total", "avg_body_length"):
        assert key in pc, f"missing policy_clauses.{key!r}"

    # Samples
    samples = report["samples"]
    assert "policy_documents" in samples
    assert "policy_clauses" in samples


def test_build_report_counts_match_seed(seeded_session) -> None:
    from backend.scripts.text_knowledge_report import build_report
    from app.seed import SCALES

    s = SCALES["small"]
    report = build_report(seeded_session)
    assert report["policy_documents"]["total"] == s["policy_documents"]
    assert report["policy_clauses"]["total"] == s["policy_clauses"]
    assert report["product_return_rules"]["total"] == s["product_return_rules"]
    assert report["product_warranty_terms"]["total"] == s["product_warranty_terms"]
    assert report["internal_agent_notes"]["total"] == s["internal_agent_notes"]
    assert report["operational_incidents"]["total"] == s["operational_incidents"]
    assert report["support_resolution_templates"]["total"] == s["support_resolution_templates"]


def test_build_report_counts_are_non_negative(seeded_session) -> None:
    from backend.scripts.text_knowledge_report import build_report

    report = build_report(seeded_session)
    assert report["policy_documents"]["total"] >= 0
    assert report["policy_clauses"]["total"] >= 0
    for k in (
        "product_return_rules",
        "product_warranty_terms",
        "internal_agent_notes",
        "operational_incidents",
        "support_resolution_templates",
    ):
        assert report[k]["total"] >= 0
    # Averages too
    assert report["policy_documents"]["avg_body_length"] >= 0
    assert report["policy_clauses"]["avg_body_length"] >= 0
    # Per-domain / per-type breakdown values must be non-negative
    for v in report["policy_documents"]["by_domain"].values():
        assert v >= 0
    for v in report["policy_documents"]["by_policy_type"].values():
        assert v >= 0


def test_build_report_avg_body_lengths_reasonable(seeded_session) -> None:
    """Policy bodies are intentionally meaningful (≥80 chars seed floor); the
    average across 50+ docs and 300+ clauses should comfortably exceed 60.
    """
    from backend.scripts.text_knowledge_report import build_report

    report = build_report(seeded_session)
    assert report["policy_documents"]["avg_body_length"] >= 100
    assert report["policy_clauses"]["avg_body_length"] >= 50


def test_build_report_samples_have_expected_shape(seeded_session) -> None:
    from backend.scripts.text_knowledge_report import build_report

    report = build_report(seeded_session)
    policy_samples = report["samples"]["policy_documents"]
    assert len(policy_samples) == 5
    for sample in policy_samples:
        assert {"id", "domain", "policy_type", "version", "title", "excerpt"} <= set(sample)
        assert isinstance(sample["excerpt"], str) and len(sample["excerpt"]) > 0

    clause_samples = report["samples"]["policy_clauses"]
    assert len(clause_samples) == 5
    for sample in clause_samples:
        assert {"id", "policy_document_id", "clause_key", "severity", "title", "excerpt"} <= set(sample)


def test_build_report_top_n_param_limits_policy_types(seeded_session) -> None:
    from backend.scripts.text_knowledge_report import build_report

    short = build_report(seeded_session, top_n_policy_types=3)
    assert len(short["policy_documents"]["by_policy_type_top_n"]) <= 3


def test_build_report_top_n_sorted_by_count_descending(seeded_session) -> None:
    from backend.scripts.text_knowledge_report import build_report

    report = build_report(seeded_session)
    counts = [row["count"] for row in report["policy_documents"]["by_policy_type_top_n"]]
    assert counts == sorted(counts, reverse=True), (
        "top_n list must be sorted by count desc"
    )


def test_build_report_domains_match_seeded_set(seeded_session) -> None:
    from backend.scripts.text_knowledge_report import build_report

    report = build_report(seeded_session)
    domains = set(report["policy_documents"]["by_domain"])
    # The 50-entry catalog covers all 5 domains.
    assert {"airline", "commerce", "saas", "support", "crm"} <= domains


# ---------------------------------------------------------------------------
# CLI: text + JSON modes
# ---------------------------------------------------------------------------


def test_cli_text_mode_runs_and_prints_sections(seeded_engine, capsys) -> None:
    engine, _ = seeded_engine
    from backend.scripts.text_knowledge_report import main

    rc = main(["--db-url", str(engine.url)])
    assert rc == 0

    out = capsys.readouterr().out
    for header in (
        "text knowledge report",
        "policy documents by domain",
        "policy documents by policy_type",
        "top",
        "sample policy excerpts",
        "sample clause excerpts",
        "total policy documents",
        "total policy clauses",
        "avg policy body length",
        "avg clause body length",
        "product return rule count",
        "warranty terms count",
        "internal agent notes count",
        "operational incidents count",
        "support resolution templates count",
    ):
        assert header in out, f"missing CLI section: {header!r}"


def test_cli_json_mode_emits_valid_json(seeded_engine, capsys) -> None:
    engine, _ = seeded_engine
    from backend.scripts.text_knowledge_report import main

    rc = main(["--db-url", str(engine.url), "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)

    # Every documented key present
    expected_keys = {
        "policy_documents",
        "policy_clauses",
        "product_return_rules",
        "product_warranty_terms",
        "internal_agent_notes",
        "operational_incidents",
        "support_resolution_templates",
        "samples",
    }
    assert expected_keys <= set(payload)

    # Counts are non-negative integers
    assert isinstance(payload["policy_documents"]["total"], int)
    assert payload["policy_documents"]["total"] >= 0
    assert isinstance(payload["policy_clauses"]["total"], int)
    assert payload["policy_clauses"]["total"] >= 0

    # Samples are present + correct length
    assert len(payload["samples"]["policy_documents"]) == 5
    assert len(payload["samples"]["policy_clauses"]) == 5


def test_cli_top_param_propagates(seeded_engine, capsys) -> None:
    engine, _ = seeded_engine
    from backend.scripts.text_knowledge_report import main

    rc = main(["--db-url", str(engine.url), "--json", "--top", "3"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["policy_documents"]["by_policy_type_top_n"]) <= 3


def test_cli_against_empty_database_returns_zeros(tmp_path) -> None:
    """Run against a fresh DB with the schema but no rows."""
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session
    from app.db import make_engine
    from app.models import Base
    from backend.scripts.text_knowledge_report import build_report

    db_path = tmp_path / "empty.db"
    engine: Engine = make_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        report = build_report(session)

    assert report["policy_documents"]["total"] == 0
    assert report["policy_clauses"]["total"] == 0
    assert report["product_return_rules"]["total"] == 0
    assert report["product_warranty_terms"]["total"] == 0
    assert report["internal_agent_notes"]["total"] == 0
    assert report["operational_incidents"]["total"] == 0
    assert report["support_resolution_templates"]["total"] == 0
    assert report["policy_documents"]["by_domain"] == {}
    assert report["policy_documents"]["by_policy_type"] == {}
    assert report["samples"]["policy_documents"] == []
    assert report["samples"]["policy_clauses"] == []
