"""PromptWall observability tables.

Phase 3A introduces ``promptwall_candidate_decisions`` — a shadow log of what
the PromptWall analyzer *would* have done, without affecting the chatbot's
behaviour. Future phases will use this table to compare predicted vs. actual
tool use across baseline and enforced modes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin
from app.models.observability import Trace


class PromptWallCandidateDecision(Base, TimestampMixin):
    __tablename__ = "promptwall_candidate_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[int] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE"), nullable=False
    )
    tool_required_predicted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_tools: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    trace: Mapped["Trace"] = relationship()

    __table_args__ = (
        Index("ix_promptwall_candidate_decisions_trace_id", "trace_id"),
    )
