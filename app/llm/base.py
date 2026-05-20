"""LLM provider interface + message/response types.

Designed to be drop-in compatible with the major providers' tool-calling
conventions (Anthropic ``tool_use``, OpenAI function calling). The shapes
exposed here are an internal lingua franca; per-provider adapters translate
to/from native payloads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class LLMToolCall(BaseModel):
    """A single tool call requested by the model."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Provider-issued call id; mock providers mint their own.")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A message in a chat conversation.

    The fields used depend on the role:

    - ``system``/``user``: ``content`` is required.
    - ``assistant``: either ``content`` (text answer) or ``tool_calls`` (tool
      requests), or both.
    - ``tool``: ``tool_call_id``, ``name``, and ``content`` (the tool result
      serialized as JSON text) are required.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[LLMToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Provider-agnostic response from one ``chat`` call."""

    model_config = ConfigDict(extra="allow")

    final_text: Optional[str] = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: int = 0
    model: str = ""
    provider: str = ""
    estimated_cost_usd: Optional[Decimal] = None
    tool_calling_supported: bool = True


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Render a list of ``ChatMessage`` to JSON-safe dicts (for logging)."""
    return [m.model_dump(mode="json", exclude_none=True) for m in messages]


class LLMProvider(ABC):
    """Abstract LLM provider. All providers must implement ``chat``."""

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Run one chat completion. Implementations must populate
        ``LLMResponse.provider``, ``model``, ``latency_ms`` and ``token_usage``.
        """
