"""TraceService: thin DB-facing wrapper over chat_sessions, traces, and tool_invocations.

Phase 1E focuses on the trace + tool_invocations rows. ``llm_calls`` will be
populated by the chatbot phase, but the schema is already in place.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import (
    ChatSession,
    LLMCall,
    PromptWallCandidateDecision,
    ToolInvocation,
    Trace,
)


class TraceService:
    """All trace-related writes for a single DB session.

    The service does NOT call ``session.commit()`` itself; the caller (a
    request handler or the ToolExecutor) decides the transaction boundary.
    This keeps the service composable inside larger units of work.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # ----- chat sessions --------------------------------------------------

    def create_chat_session(
        self,
        *,
        customer_id: Optional[int] = None,
        channel: str = "web",
        session_uuid: Optional[str] = None,
    ) -> ChatSession:
        chat = ChatSession(
            session_uuid=session_uuid or uuid.uuid4().hex,
            customer_id=customer_id,
            channel=channel,
        )
        self.session.add(chat)
        self.session.flush()
        return chat

    # ----- traces ---------------------------------------------------------

    def create_trace(
        self,
        *,
        session_id: int,
        user_message: str,
        mode: str = "baseline",
        customer_id: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Trace:
        trace = Trace(
            session_id=session_id,
            customer_id=customer_id,
            mode=mode,
            user_message=user_message,
            extra_metadata=metadata,
        )
        self.session.add(trace)
        self.session.flush()
        return trace

    def finish_trace(
        self,
        trace_id: int,
        *,
        final_answer: Optional[str] = None,
        latency_ms: Optional[int] = None,
    ) -> Trace:
        trace = self.session.get(Trace, trace_id)
        if trace is None:
            raise ValueError(f"trace {trace_id} not found")
        trace.final_answer = final_answer
        trace.ended_at = datetime.now(timezone.utc)
        if latency_ms is None and trace.started_at is not None:
            started = trace.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            latency_ms = int((trace.ended_at - started).total_seconds() * 1000)
        trace.latency_ms = latency_ms
        self.session.flush()
        return trace

    # ----- tool invocations ----------------------------------------------

    def log_llm_call(
        self,
        *,
        trace_id: int,
        provider: str,
        model: str,
        input_messages: list[dict[str, Any]],
        output_message: Optional[str],
        tool_calls_requested: Optional[list[dict[str, Any]]],
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        estimated_cost_usd: Optional[Decimal] = None,
        latency_ms: Optional[int] = None,
    ) -> LLMCall:
        call = LLMCall(
            trace_id=trace_id,
            provider=provider,
            model=model,
            input_messages=input_messages,
            output_message=output_message,
            tool_calls_requested=tool_calls_requested,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            latency_ms=latency_ms,
        )
        self.session.add(call)
        self.session.flush()
        return call

    def log_candidate_decision(
        self,
        *,
        trace_id: int,
        tool_required_predicted: bool,
        predicted_tools: list[str],
        confidence: float,
        reason: Optional[str] = None,
    ) -> PromptWallCandidateDecision:
        """Persist a PromptWall shadow decision into ``promptwall_candidate_decisions``."""
        decision = PromptWallCandidateDecision(
            trace_id=trace_id,
            tool_required_predicted=tool_required_predicted,
            predicted_tools=list(predicted_tools),
            confidence=float(confidence),
            reason=reason,
        )
        self.session.add(decision)
        self.session.flush()
        return decision

    def log_tool_invocation(
        self,
        *,
        trace_id: int,
        tool_name: str,
        input_json: Optional[dict[str, Any]],
        output_json: Optional[dict[str, Any]],
        success: bool,
        latency_ms: int,
        error_message: Optional[str] = None,
        evidence_id: Optional[str] = None,
    ) -> ToolInvocation:
        invocation = ToolInvocation(
            trace_id=trace_id,
            tool_name=tool_name,
            input_json=input_json,
            output_json=output_json,
            success=success,
            error_message=error_message,
            latency_ms=latency_ms,
            evidence_id=evidence_id,
        )
        self.session.add(invocation)
        self.session.flush()
        return invocation
