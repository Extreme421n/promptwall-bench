"""Tool primitive, registry, exceptions, and invocation result.

A ``Tool`` bundles its metadata (used by both the future LLM function-calling
layer and the future PromptWall policy layer) with a typed implementation.

The contract is intentionally minimal:

    output = tool.call(session, raw_input_dict)

``call`` validates input via the tool's Pydantic schema, executes the
implementation, validates the output, and returns a plain ``dict``. Errors
raise typed exceptions so callers can distinguish validation, not-found,
ambiguity, and unexpected failures.

``invoke_tool`` is the trace-ready entry point: it never raises and returns
an ``InvocationResult`` with success/error metadata + latency. This is the
shape the future trace logger will persist into ``tool_invocations``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base class for all tool errors."""


class ToolValidationError(ToolError):
    """Input does not match the tool's declared input schema."""


class ToolNotFoundError(ToolError):
    """A tool with the requested name is not registered."""


class ResourceNotFoundError(ToolError):
    """The tool ran but the requested entity does not exist."""


class AmbiguousInputError(ToolError):
    """The input is valid but matches multiple entities and cannot be resolved."""


RiskLevel = Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


ToolImpl = Callable[[Session, BaseModel], BaseModel]


@dataclass(frozen=True)
class Tool:
    """A registered tool callable from the chatbot (in a later phase)."""

    name: str
    description: str
    domain: str  # e.g. "crm", "airline", "support", "kb"
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    risk_level: RiskLevel
    read_only: bool
    impl: ToolImpl

    def call(self, session: Session, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Validate, execute, and return a JSON-safe dict.

        Raises:
            ToolValidationError: input doesn't match the schema
            ResourceNotFoundError: entity not found
            AmbiguousInputError: input matches multiple entities ambiguously
            ToolError: any other tool-level failure
        """
        try:
            validated = self.input_schema.model_validate(raw_input)
        except ValidationError as e:
            raise ToolValidationError(_format_validation_error(e)) from e

        result = self.impl(session, validated)
        # Defensive: ensure impl returned the right shape.
        if not isinstance(result, self.output_schema):
            raise ToolError(
                f"tool {self.name!r} returned {type(result).__name__}, "
                f"expected {self.output_schema.__name__}"
            )
        return result.model_dump(mode="json")

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable description suitable for an LLM function-calling spec."""
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "risk_level": self.risk_level,
            "read_only": self.read_only,
            "input_schema": self.input_schema.model_json_schema(),
            "output_schema": self.output_schema.model_json_schema(),
        }


def _format_validation_error(e: ValidationError) -> str:
    parts: list[str] = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', '')}")
    return "; ".join(parts) if parts else str(e)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    """Mutable map of tool name -> Tool."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as e:
            raise ToolNotFoundError(f"no tool named {name!r}") from e

    def list_tools(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe_all(self) -> list[dict[str, Any]]:
        return [t.describe() for t in self.list_tools()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools


default_registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Trace-ready invocation entry point
# ---------------------------------------------------------------------------


@dataclass
class InvocationResult:
    """Trace-ready result of a tool invocation.

    Phase 1D returns this from the in-memory ``invoke_tool`` (no DB writes).
    Phase 1E persists invocations via ``ToolExecutor`` and populates the
    ``evidence_id`` field, which links the result to a row in
    ``tool_invocations``.
    """

    tool_name: str
    success: bool
    output: dict[str, Any] | None
    error_type: str | None
    error_message: str | None
    latency_ms: int
    evidence_id: str | None = None


def invoke_tool(
    name: str,
    raw_input: dict[str, Any],
    session: Session,
    registry: ToolRegistry | None = None,
) -> InvocationResult:
    """Trace-ready, never-raises wrapper around ``Tool.call``."""
    reg = registry or default_registry
    t0 = time.perf_counter()

    try:
        tool = reg.get(name)
    except ToolNotFoundError as e:
        return InvocationResult(
            tool_name=name,
            success=False,
            output=None,
            error_type="ToolNotFoundError",
            error_message=str(e),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        output = tool.call(session, raw_input)
        return InvocationResult(
            tool_name=name,
            success=True,
            output=output,
            error_type=None,
            error_message=None,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except ToolError as e:
        return InvocationResult(
            tool_name=name,
            success=False,
            output=None,
            error_type=type(e).__name__,
            error_message=str(e),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
    except Exception as e:  # noqa: BLE001 - intentional catch-all for trace fidelity
        return InvocationResult(
            tool_name=name,
            success=False,
            output=None,
            error_type=type(e).__name__,
            error_message=f"unexpected error: {e}",
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
