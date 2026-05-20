"""Observability tables for chat sessions, traces, LLM calls, and tool invocations.

These are kept intentionally lightweight in Phase 1B. They will be exercised
once the chatbot phase is added.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base, TimestampMixin
from app.models.crm import Customer


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_uuid: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    customer_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"), nullable=True
    )
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="web")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    customer: Mapped[Optional["Customer"]] = relationship()
    traces: Mapped[list["Trace"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_chat_sessions_customer_id", "customer_id"),
    )


class Trace(Base, TimestampMixin):
    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    customer_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"), nullable=True
    )
    mode: Mapped[str] = mapped_column(String(40), nullable=False, default="baseline")
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # ``metadata`` is reserved on DeclarativeBase, so we map the column name
    # explicitly while exposing it as ``extra_metadata`` in Python.
    extra_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSON, nullable=True
    )

    session: Mapped["ChatSession"] = relationship(back_populates="traces")
    customer: Mapped[Optional["Customer"]] = relationship(back_populates="traces")
    llm_calls: Mapped[list["LLMCall"]] = relationship(
        back_populates="trace", cascade="all, delete-orphan"
    )
    tool_invocations: Mapped[list["ToolInvocation"]] = relationship(
        back_populates="trace", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_traces_session_id", "session_id"),
        Index("ix_traces_customer_id", "customer_id"),
        Index("ix_traces_mode", "mode"),
    )


class LLMCall(Base, TimestampMixin):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[int] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    input_messages: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    output_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tool_calls_requested: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(
        JSON, nullable=True
    )
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 6), nullable=True
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    trace: Mapped["Trace"] = relationship(back_populates="llm_calls")

    __table_args__ = (
        Index("ix_llm_calls_trace_id", "trace_id"),
    )


class ToolInvocation(Base, TimestampMixin):
    __tablename__ = "tool_invocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[int] = mapped_column(
        ForeignKey("traces.id", ondelete="CASCADE"), nullable=False
    )
    tool_name: Mapped[str] = mapped_column(String(80), nullable=False)
    input_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    output_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    trace: Mapped["Trace"] = relationship(back_populates="tool_invocations")

    __table_args__ = (
        Index("ix_tool_invocations_trace_id", "trace_id"),
        Index("ix_tool_invocations_tool_name", "tool_name"),
    )
