"""Evaluation tables: one row per benchmark run and one per scored case."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin
from app.models.observability import Trace


class EvaluationRun(Base, TimestampMixin):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    mode: Mapped[str] = mapped_column(String(40), nullable=False, default="baseline")
    model: Mapped[str] = mapped_column(String(80), nullable=False, default="mock")
    eval_file: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metrics_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    results: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_evaluation_runs_mode", "mode"),
    )


class EvaluationResult(Base, TimestampMixin):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    case_id: Mapped[str] = mapped_column(String(40), nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    # Phase E1 — store the case's declared domain + risk so per-domain and
    # per-risk metrics can be computed without re-loading the JSONL. Nullable
    # for backwards compatibility with results written before Phase E1.
    expected_domain: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    risk: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    trace_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("traces.id", ondelete="SET NULL"), nullable=True
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    must_use_tool: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_tools_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    actual_tools_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    tool_called_when_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    tool_skip: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    expected_tool_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    wrong_tool: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    missing_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    clarification_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    suspicious_unsupported_claim: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped["EvaluationRun"] = relationship(back_populates="results")
    trace: Mapped[Optional["Trace"]] = relationship()

    __table_args__ = (
        Index("ix_evaluation_results_run_id", "run_id"),
        Index("ix_evaluation_results_case_id", "case_id"),
    )
