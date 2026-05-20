"""Tests for the SCALES presets and the db_size CLI.

We never run medium or large seeds in CI — they take long enough that they'd
dominate the suite. Instead we assert the presets are defined correctly and
that ``db_size`` reports against the small seed produced by the shared fixture.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import Engine

from app.seed import SCALES


# ---------------------------------------------------------------------------
# SCALES
# ---------------------------------------------------------------------------


def test_three_scales_defined() -> None:
    assert set(SCALES) == {"small", "medium", "large"}


def test_small_preset_matches_phase_1c_targets() -> None:
    assert SCALES["small"]["customers"] == 500
    assert SCALES["small"]["bookings"] == 1000
    assert SCALES["small"]["seats"] == 3000
    assert SCALES["small"]["support_messages"] == 2500
    assert SCALES["small"]["kb_articles"] == 50


def test_medium_preset_matches_phase_2d_targets() -> None:
    m = SCALES["medium"]
    assert m["customers"] == 20_000
    assert m["flights"] == 10_000
    assert m["bookings"] == 50_000
    assert m["seats"] == 200_000
    assert m["refunds"] == 15_000
    assert m["support_tickets"] == 50_000
    assert m["support_messages"] == 250_000
    assert m["kb_articles"] == 1_000


def test_large_preset_strictly_larger_than_medium() -> None:
    """Sanity check the monotonic scale increase."""
    for key in (
        "customers",
        "flights",
        "bookings",
        "seats",
        "refunds",
        "support_tickets",
        "support_messages",
        "kb_articles",
    ):
        assert SCALES["large"][key] >= SCALES["medium"][key], key
        assert SCALES["medium"][key] >= SCALES["small"][key], key


def test_large_targets_one_gb_via_messages() -> None:
    """The large preset's support_messages count should be enough to clear ~1GB
    on disk given realistic body lengths (we keep the assertion conservative).
    """
    assert SCALES["large"]["support_messages"] >= 1_000_000


def test_seed_rejects_unknown_scale() -> None:
    from app.seed import seed

    # Use a throwaway in-memory engine so we don't touch any real DB.
    from app.db import make_engine

    engine = make_engine("sqlite:///:memory:")
    import pytest

    with pytest.raises(ValueError, match="unknown scale"):
        seed(engine, scale="xxxlarge")


# ---------------------------------------------------------------------------
# db_size CLI
# ---------------------------------------------------------------------------


def test_db_size_cli_reports_against_seeded_db(seeded_engine, capsys) -> None:
    engine, _ = seeded_engine
    from backend.scripts.db_size import main

    rc = main(["--db-url", str(engine.url)])
    assert rc == 0
    out = capsys.readouterr().out
    # Header + total size line + row count for one known table at least.
    assert "database url" in out
    assert "total size" in out
    assert "customers" in out
    assert "support_messages" in out
    assert "total rows" in out


def test_db_size_format_helpers() -> None:
    from backend.scripts.db_size import _format_bytes, _sqlite_file_path

    assert _format_bytes(0) == "0.00 B"
    assert _format_bytes(1024) == "1.00 KB"
    assert _format_bytes(1024 * 1024) == "1.00 MB"
    assert _format_bytes(1024 * 1024 * 1024) == "1.00 GB"
    assert _sqlite_file_path("sqlite:///./bench.db") == Path("./bench.db")
    assert _sqlite_file_path("sqlite:///:memory:") is None
    assert _sqlite_file_path("postgresql+psycopg://x/y") is None


def test_db_size_includes_total_size_number_when_file_exists(
    seeded_engine, capsys
) -> None:
    engine, _ = seeded_engine
    from backend.scripts.db_size import main

    main(["--db-url", str(engine.url)])
    out = capsys.readouterr().out
    # Either we see a unit (B/KB/MB/...) or the "unknown" fallback.
    assert any(unit in out for unit in (" B (", " KB (", " MB (", " GB ("))


# ---------------------------------------------------------------------------
# Smoke test that the small preset's invariants survive the bulk-insert rewrite
# ---------------------------------------------------------------------------


def test_small_seed_supports_ambiguous_pairs(seeded_engine) -> None:
    """The 50 deliberately-ambiguous TKT-<pnr>/<pnr> pairs survive the rewrite."""
    engine, summary = seeded_engine
    assert summary["customers"] == 500
    assert summary["support_messages"] == 2500

    from sqlalchemy import func, select
    from sqlalchemy.orm import Session
    from app.models import Booking, SupportTicket

    with Session(engine) as s:
        matches = s.execute(
            select(SupportTicket.ticket_number, Booking.booking_reference)
            .join(
                Booking,
                (Booking.customer_id == SupportTicket.customer_id)
                & (Booking.booking_reference == func.substr(SupportTicket.ticket_number, 5)),
            )
        ).all()
    assert len(matches) >= 50
