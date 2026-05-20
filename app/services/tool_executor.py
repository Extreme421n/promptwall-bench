"""ToolExecutor: run a tool via the registry, time it, and log it.

This is the bridge between Phase 1D's pure tool layer and the observability
tables added in Phase 1B. Every successful invocation gets a fresh
``evidence_id`` that the chatbot (later phase) can cite when forming
grounded answers.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.services.trace_service import TraceService
from app.tools.base import (
    InvocationResult,
    Tool,
    ToolError,
    ToolNotFoundError,
    ToolRegistry,
    default_registry,
)


class ToolExecutor:
    """Execute tools and persist their invocations.

    Usage:
        executor = ToolExecutor(session)
        result = executor.execute_tool(trace_id, "get_customer_profile", {"customer_id": 7})

    If ``trace_id`` is None, a synthetic chat session + trace are created so
    that the invocation can still be persisted. This is what the
    ``POST /tools/{name}/execute`` endpoint uses when no trace context is
    provided.
    """

    def __init__(
        self,
        session: Session,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.session = session
        self.registry = registry or default_registry
        self.trace_service = TraceService(session)

    # ----- public API -----------------------------------------------------

    def execute_tool(
        self,
        trace_id: Optional[int],
        tool_name: str,
        input_json: dict[str, Any],
    ) -> InvocationResult:
        resolved_trace_id = trace_id if trace_id is not None else self._create_temp_trace(tool_name)

        t0 = time.perf_counter()

        # 1) Resolve the tool. A missing tool is a logged failure too.
        try:
            tool: Tool = self.registry.get(tool_name)
        except ToolNotFoundError as e:
            latency_ms = self._elapsed_ms(t0)
            self.trace_service.log_tool_invocation(
                trace_id=resolved_trace_id,
                tool_name=tool_name,
                input_json=input_json,
                output_json=None,
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
                evidence_id=None,
            )
            self.session.commit()
            return InvocationResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error_type="ToolNotFoundError",
                error_message=str(e),
                latency_ms=latency_ms,
                evidence_id=None,
            )

        # 2) Execute the tool.
        try:
            output = tool.call(self.session, input_json)
        except ToolError as e:
            latency_ms = self._elapsed_ms(t0)
            self.trace_service.log_tool_invocation(
                trace_id=resolved_trace_id,
                tool_name=tool_name,
                input_json=input_json,
                output_json=None,
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
                evidence_id=None,
            )
            self.session.commit()
            return InvocationResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error_type=type(e).__name__,
                error_message=str(e),
                latency_ms=latency_ms,
                evidence_id=None,
            )
        except Exception as e:  # noqa: BLE001 — capture unexpected failures into the trace
            latency_ms = self._elapsed_ms(t0)
            # The tool's own SELECT may have left the session in a bad state.
            self.session.rollback()
            self.trace_service.log_tool_invocation(
                trace_id=resolved_trace_id,
                tool_name=tool_name,
                input_json=input_json,
                output_json=None,
                success=False,
                latency_ms=latency_ms,
                error_message=f"unexpected error: {e}",
                evidence_id=None,
            )
            self.session.commit()
            return InvocationResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error_type=type(e).__name__,
                error_message=f"unexpected error: {e}",
                latency_ms=latency_ms,
                evidence_id=None,
            )

        # 3) Success path: mint evidence, log, commit.
        latency_ms = self._elapsed_ms(t0)
        evidence_id = f"ev_{uuid.uuid4().hex}"
        self.trace_service.log_tool_invocation(
            trace_id=resolved_trace_id,
            tool_name=tool_name,
            input_json=input_json,
            output_json=output,
            success=True,
            latency_ms=latency_ms,
            error_message=None,
            evidence_id=evidence_id,
        )
        self.session.commit()
        return InvocationResult(
            tool_name=tool_name,
            success=True,
            output=output,
            error_type=None,
            error_message=None,
            latency_ms=latency_ms,
            evidence_id=evidence_id,
        )

    # ----- helpers --------------------------------------------------------

    def _create_temp_trace(self, tool_name: str) -> int:
        """Create a one-off chat session + trace for an untraced tool call."""
        chat = self.trace_service.create_chat_session(channel="tool_test")
        trace = self.trace_service.create_trace(
            session_id=chat.id,
            user_message=f"[temp trace for tool={tool_name}]",
            mode="tool_test",
            metadata={"temp": True, "tool": tool_name},
        )
        self.session.commit()
        return trace.id

    @staticmethod
    def _elapsed_ms(t0: float) -> int:
        return int((time.perf_counter() - t0) * 1000)
