"""ChatService: drives the baseline chatbot loop and writes every trace artifact.

Baseline loop:

    1. Build [system, user] messages and the tool spec list from the registry.
    2. Call the LLM provider. Log the LLM call.
    3. If the response has tool calls, execute each via ToolExecutor and
       append a ``tool`` message to the conversation. Loop back to step 2.
    4. When the LLM returns final text (no tool calls), finish the trace.

Phase 1G uses the MockLLMProvider by default. Adding a real provider in a
later phase requires no changes to this service.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

import uuid

from app.llm import (
    ChatMessage,
    LLMProvider,
    LLMToolCall,
    messages_to_dicts,
)
from app.models import ChatSession
from app.promptwall import (
    EnforcementDecision,
    PromptWallCandidateAnalyzer,
    PromptWallRouter,
)
from app.services.tool_executor import ToolExecutor
from app.services.trace_service import TraceService
from app.tools import default_registry
from app.tools.base import ToolRegistry

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful airline customer support assistant. You have access to "
    "tools. Use tools whenever the user asks about customer-specific, "
    "booking-specific, flight-specific, refund-specific, ticket-specific, "
    "baggage policy, or dynamic operational information. If required "
    "information is missing, ask a clarification question. Do not invent IDs, "
    "prices, dates, statuses, refund amounts, gates, or policy details."
)

# Modes that should produce identical behaviour to baseline. The chat service
# treats them the same; the only thing that changes is which observability
# rows are written (PromptWall shadow modes also log a candidate decision).
_BASELINE_LIKE_MODES = {"baseline", "promptwall_candidate_shadow"}

_ENFORCED_INSTRUCTION = (
    "Answer using only the verified evidence above. If the evidence is "
    "insufficient to answer fully, say what information is missing rather "
    "than guessing. Do not call the tool that already produced the evidence."
)

# Upper bound on the loop, in case a provider keeps requesting tools forever.
_MAX_TURNS = 4


@dataclass
class ToolCallTrace:
    name: str
    arguments: dict[str, Any]
    success: bool
    evidence_id: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    latency_ms: int


@dataclass
class ChatResult:
    answer: str
    trace_id: int
    session_uuid: str
    tools_called: list[ToolCallTrace] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    estimated_cost_usd: Decimal = Decimal("0")


class ChatService:
    """Top-level orchestrator for one /chat request."""

    def __init__(
        self,
        session: Session,
        provider: LLMProvider,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.session = session
        self.provider = provider
        self.registry = registry or default_registry
        self.trace_service = TraceService(session)
        self.executor = ToolExecutor(session, self.registry)

    # ------------------------------------------------------------------

    def chat(
        self,
        *,
        message: str,
        session_uuid: Optional[str] = None,
        customer_id: Optional[int] = None,
        mode: str = "baseline",
        metadata: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> ChatResult:
        t0 = time.perf_counter()

        chat = self._resolve_chat_session(session_uuid, customer_id)
        trace = self.trace_service.create_trace(
            session_id=chat.id,
            user_message=message,
            mode=mode,
            customer_id=customer_id,
            metadata=metadata,
        )
        self.session.commit()

        # Phase 3A — PromptWall shadow analysis. The decision is logged but
        # never affects the chat loop below. Any failure here is swallowed so
        # the baseline path is never disturbed.
        if mode in ("promptwall_candidate_shadow", "promptwall_enforced"):
            self._record_candidate_decision(trace_id=trace.id, message=message, metadata=metadata)

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=BASELINE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=message),
        ]
        tool_specs = self.registry.describe_all()

        tools_called: list[ToolCallTrace] = []
        evidence_ids: list[str] = []
        total_cost = Decimal("0")
        final_answer = ""

        # Phase 4A — PromptWall enforcement. The router pre-executes a tool
        # before the LLM call when it has high confidence. The result is
        # injected into the conversation as a completed tool call so the LLM
        # only has to summarise the answer; we also remove the forced tool's
        # spec from the offered list to prevent a redundant re-call.
        if mode == "promptwall_enforced":
            forced_tool = self._maybe_enforce(
                trace_id=trace.id,
                user_message=message,
                messages=messages,
                tool_specs=tool_specs,
                tools_called=tools_called,
                evidence_ids=evidence_ids,
            )
            if forced_tool is not None:
                # Filter out the spec so the LLM doesn't try to re-call.
                tool_specs = [t for t in tool_specs if t["name"] != forced_tool]

        for _ in range(_MAX_TURNS):
            llm_response = self.provider.chat(messages, tools=tool_specs, model=model)

            # Persist this LLM turn.
            self.trace_service.log_llm_call(
                trace_id=trace.id,
                provider=llm_response.provider,
                model=llm_response.model,
                input_messages=messages_to_dicts(messages),
                output_message=llm_response.final_text,
                tool_calls_requested=(
                    [tc.model_dump(mode="json") for tc in llm_response.tool_calls]
                    if llm_response.tool_calls
                    else None
                ),
                prompt_tokens=llm_response.token_usage.prompt_tokens,
                completion_tokens=llm_response.token_usage.completion_tokens,
                total_tokens=llm_response.token_usage.total_tokens,
                estimated_cost_usd=llm_response.estimated_cost_usd,
                latency_ms=llm_response.latency_ms,
            )
            if llm_response.estimated_cost_usd is not None:
                total_cost += llm_response.estimated_cost_usd
            self.session.commit()

            if not llm_response.tool_calls:
                final_answer = llm_response.final_text or ""
                break

            # Record the assistant turn that requested the tools.
            messages.append(
                ChatMessage(role="assistant", tool_calls=list(llm_response.tool_calls))
            )

            # Execute each requested tool, appending its result.
            for tc in llm_response.tool_calls:
                inv = self.executor.execute_tool(
                    trace_id=trace.id, tool_name=tc.name, input_json=tc.arguments
                )
                tools_called.append(
                    ToolCallTrace(
                        name=tc.name,
                        arguments=tc.arguments,
                        success=inv.success,
                        evidence_id=inv.evidence_id,
                        error_type=inv.error_type,
                        error_message=inv.error_message,
                        latency_ms=inv.latency_ms,
                    )
                )
                if inv.evidence_id:
                    evidence_ids.append(inv.evidence_id)

                tool_payload = (
                    inv.output
                    if inv.success
                    else {"error": inv.error_message, "error_type": inv.error_type}
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=json.dumps(tool_payload, default=str),
                    )
                )
        else:
            # Loop exhausted without a final answer; surface a graceful fallback.
            final_answer = (
                "I wasn't able to produce a final answer in the allotted turns. "
                "Please try rephrasing or providing more specific information."
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        self.trace_service.finish_trace(
            trace.id, final_answer=final_answer, latency_ms=latency_ms
        )
        self.session.commit()

        return ChatResult(
            answer=final_answer,
            trace_id=trace.id,
            session_uuid=chat.session_uuid,
            tools_called=tools_called,
            evidence_ids=evidence_ids,
            latency_ms=latency_ms,
            estimated_cost_usd=total_cost,
        )

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PromptWall enforcement (Phase 4A)
    # ------------------------------------------------------------------

    def _maybe_enforce(
        self,
        *,
        trace_id: int,
        user_message: str,
        messages: list[ChatMessage],
        tool_specs: list[dict[str, Any]],
        tools_called: list[ToolCallTrace],
        evidence_ids: list[str],
    ) -> Optional[str]:
        """If the router decides to enforce, pre-execute the tool and seed the
        conversation with the evidence. Returns the forced tool name (or None)."""
        router = PromptWallRouter()
        decision = router.decide(
            message=user_message,
            available_tools=set(t["name"] for t in tool_specs),
        )
        if not decision.should_enforce or decision.tool_name is None:
            return None

        # Run the forced tool through the same path as any other invocation
        # so it lands in tool_invocations with a fresh evidence_id (when it
        # succeeds).
        inv = self.executor.execute_tool(
            trace_id=trace_id,
            tool_name=decision.tool_name,
            input_json=decision.arguments,
        )
        tools_called.append(
            ToolCallTrace(
                name=decision.tool_name,
                arguments=decision.arguments,
                success=inv.success,
                evidence_id=inv.evidence_id,
                error_type=inv.error_type,
                error_message=inv.error_message,
                latency_ms=inv.latency_ms,
            )
        )
        if inv.evidence_id:
            evidence_ids.append(inv.evidence_id)

        evidence_text = self._format_evidence_block(
            tool_name=decision.tool_name,
            evidence_id=inv.evidence_id,
            output=inv.output if inv.success else None,
            error_message=inv.error_message if not inv.success else None,
            confidence=decision.confidence,
            reason=decision.reason,
        )

        # System messages: a copy of the evidence (for human-readable LLMs)
        # and a strict instruction. We insert AFTER the baseline system prompt
        # so providers that only honour the first system message still see the
        # full one.
        messages.insert(1, ChatMessage(role="system", content=evidence_text))
        messages.insert(2, ChatMessage(role="system", content=_ENFORCED_INSTRUCTION))

        # Simulate a tool call so providers (mock + real) see a completed
        # turn and just need to produce the final answer.
        forced_call_id = f"call_pw_{uuid.uuid4().hex[:12]}"
        messages.append(
            ChatMessage(
                role="assistant",
                tool_calls=[
                    LLMToolCall(
                        id=forced_call_id,
                        name=decision.tool_name,
                        arguments=decision.arguments,
                    )
                ],
            )
        )
        tool_payload: dict[str, Any] = (
            inv.output if inv.success else {"error": inv.error_message, "error_type": inv.error_type}
        ) or {}
        messages.append(
            ChatMessage(
                role="tool",
                tool_call_id=forced_call_id,
                name=decision.tool_name,
                content=json.dumps(tool_payload, default=str),
            )
        )
        return decision.tool_name

    @staticmethod
    def _format_evidence_block(
        *,
        tool_name: str,
        evidence_id: Optional[str],
        output: Optional[dict[str, Any]],
        error_message: Optional[str],
        confidence: float,
        reason: str,
    ) -> str:
        lines = [
            "PromptWall verified evidence:",
            f"- evidence_id: {evidence_id or '(none)'}",
            f"- tool_name: {tool_name}",
            f"- router_confidence: {confidence:.2f}",
            f"- router_reason: {reason}",
        ]
        if output is None:
            lines.append(f"- status: failed — {error_message or 'tool returned no data'}")
            return "\n".join(lines)

        # Render up to ~8 top-level fields. If the tool returned a {count, items: [...]}
        # shape (most do), summarise the first 1-2 items.
        count = output.get("count")
        if count is not None:
            lines.append(f"- count: {count}")
        items_key = next(
            (k for k in ("bookings", "flights", "refunds", "policies", "tickets", "seats") if k in output),
            None,
        )
        if items_key:
            items = output.get(items_key) or []
            for i, item in enumerate(items[:2]):
                lines.append(f"- {items_key}[{i}]:")
                for k, v in list(item.items())[:8]:
                    lines.append(f"    {k}: {v}")
            if len(items) > 2:
                lines.append(f"  ... +{len(items) - 2} more")
        else:
            for k, v in list(output.items())[:8]:
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)

    def _record_candidate_decision(
        self, *, trace_id: int, message: str, metadata: Optional[dict[str, Any]]
    ) -> None:
        """Shadow analysis: run the analyzer and persist its decision."""
        try:
            analyzer = PromptWallCandidateAnalyzer()
            decision = analyzer.analyze(
                message=message,
                available_tools=self.registry.names(),
                context=metadata or None,
            )
            self.trace_service.log_candidate_decision(
                trace_id=trace_id,
                tool_required_predicted=decision.tool_required_predicted,
                predicted_tools=decision.predicted_tools,
                confidence=decision.confidence,
                reason=decision.reason,
            )
            self.session.commit()
        except Exception:  # noqa: BLE001 — shadow analysis must never break the chat path
            self.session.rollback()

    def _resolve_chat_session(
        self, session_uuid: Optional[str], customer_id: Optional[int]
    ) -> ChatSession:
        if session_uuid:
            existing = self.session.execute(
                select(ChatSession).where(ChatSession.session_uuid == session_uuid)
            ).scalar_one_or_none()
            if existing is not None:
                return existing
            return self.trace_service.create_chat_session(
                customer_id=customer_id, session_uuid=session_uuid
            )
        return self.trace_service.create_chat_session(customer_id=customer_id)
