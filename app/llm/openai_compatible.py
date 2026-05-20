"""OpenAI-compatible LLM provider.

Targets the OpenAI Chat Completions API and any compatible provider exposed
through a custom ``base_url`` (vLLM, Groq, Together, etc.).

Highlights:

- Tool/function calling via the ``tools`` parameter; gracefully falls back to
  text-only if the model rejects the tool spec.
- Reports prompt/completion/total tokens and (best-effort) USD cost using a
  small built-in pricing table.
- The OpenAI SDK client can be injected at construction time so tests don't
  need a real API key or network.
"""

from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from app.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Pricing per 1M tokens, USD. Numbers are illustrative; chatbot benchmarks
# tolerate approximate cost reporting. Unknown models report cost=None.
_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-2024-08-06": {"prompt": 2.50, "completion": 10.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
}


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[Decimal]:
    table = _PRICING_USD_PER_1M.get(model)
    if not table:
        # Strip date suffixes like '-2024-08-06' before giving up.
        base = model.rsplit("-", 3)[0] if model.count("-") >= 3 else model
        table = _PRICING_USD_PER_1M.get(base)
    if not table:
        return None
    cost = (
        Decimal(prompt_tokens) * Decimal(str(table["prompt"]))
        + Decimal(completion_tokens) * Decimal(str(table["completion"]))
    ) / Decimal("1000000")
    return cost.quantize(Decimal("0.000001"))


class OpenAICompatibleProvider(LLMProvider):
    """Thin wrapper around ``openai.OpenAI().chat.completions.create``."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Any = None,
        supports_tools: bool = True,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if client is None:
            if not api_key:
                raise ValueError(
                    "openai_api_key is required for OpenAICompatibleProvider when no client is injected"
                )
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - import guard
                raise ImportError(
                    "openai SDK is required; install with `pip install openai`"
                ) from e
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self._model = model
        self._supports_tools = supports_tools

    @property
    def default_model(self) -> str:  # type: ignore[override]
        return self._model

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        chosen_model = model or self._model

        kwargs: dict[str, Any] = {
            "model": chosen_model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }

        send_tools = bool(tools) and self._supports_tools
        if send_tools:
            kwargs["tools"] = [self._convert_tool_spec(t) for t in tools or []]

        try:
            raw = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - we triage by error text
            if send_tools and _looks_like_tool_unsupported(e):
                logger.warning(
                    "model %s rejected tool calling (%s); retrying without tools",
                    chosen_model,
                    e,
                )
                self._supports_tools = False
                kwargs.pop("tools", None)
                raw = self._client.chat.completions.create(**kwargs)
            else:
                raise

        return self._parse_response(raw, chosen_model, t0)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.tool_call_id or "",
                        "content": m.content or "",
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    @staticmethod
    def _convert_tool_spec(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: Any, model: str, t0: float) -> LLMResponse:
        choices = getattr(raw, "choices", None) or []
        if not choices:
            raise ValueError("LLM response contained no choices")
        msg = choices[0].message

        tool_calls: list[LLMToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            raw_args = getattr(fn, "arguments", "") if fn else ""
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    args = {"_raw_arguments": raw_args}
            except json.JSONDecodeError:
                args = {"_raw_arguments": raw_args}
            tool_calls.append(
                LLMToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=(getattr(fn, "name", "") if fn else "") or "",
                    arguments=args,
                )
            )

        content = getattr(msg, "content", None)
        usage = getattr(raw, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            final_text=content,
            tool_calls=tool_calls,
            raw_response={
                "id": getattr(raw, "id", None),
                "model": getattr(raw, "model", model),
                "object": getattr(raw, "object", "chat.completion"),
                "finish_reason": getattr(choices[0], "finish_reason", None),
            },
            token_usage=token_usage,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            model=model,
            provider=self.name,
            estimated_cost_usd=_estimate_cost_usd(model, prompt_tokens, completion_tokens),
            tool_calling_supported=self._supports_tools,
        )


def _looks_like_tool_unsupported(exc: BaseException) -> bool:
    """Heuristic: did this error come from a model rejecting the tool spec?"""
    text = (str(exc) or "").lower()
    return any(
        kw in text
        for kw in (
            "does not support tool",
            "does not support function",
            "tool_choice",
            "tool_calls is not supported",
            "function call",
            "tool calling",
            "tools is not supported",
            "unsupported parameter: 'tools'",
        )
    )
