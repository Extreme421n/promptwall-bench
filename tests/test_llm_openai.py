"""Tests for OpenAICompatibleProvider and the get_provider factory.

These tests never make a network call. We inject a stand-in client whose
``chat.completions.create`` returns simple namespaced objects matching the
shape the openai SDK produces.
"""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from app.config import settings
from app.llm import (
    ChatMessage,
    MockLLMProvider,
    OpenAICompatibleProvider,
    get_provider,
)
from app.llm.openai_compatible import _estimate_cost_usd


# ---------------------------------------------------------------------------
# Test doubles for the openai SDK response shape
# ---------------------------------------------------------------------------


def _make_completion(
    *,
    content: Optional[str] = None,
    tool_calls: Optional[list[dict[str, Any]]] = None,
    prompt: int = 12,
    completion: int = 7,
    model: str = "gpt-4o-mini",
) -> SimpleNamespace:
    sdk_tool_calls = None
    if tool_calls:
        sdk_tool_calls = [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"], arguments=json.dumps(tc["arguments"])
                ),
            )
            for tc in tool_calls
        ]
    message = SimpleNamespace(content=content, tool_calls=sdk_tool_calls)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )
    return SimpleNamespace(
        id="cmpl-test-1", model=model, object="chat.completion",
        choices=[choice], usage=usage,
    )


class _FakeOpenAIClient:
    """Stand-in for ``openai.OpenAI()`` that records calls and returns canned completions."""

    def __init__(self, completions: list[Any], raise_first_with: Optional[Exception] = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._completions = list(completions)
        self._raise_first_with = raise_first_with
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raise_first_with is not None:
            err = self._raise_first_with
            self._raise_first_with = None
            raise err
        return self._completions.pop(0)


# ---------------------------------------------------------------------------
# get_provider factory
# ---------------------------------------------------------------------------


def test_get_provider_returns_mock_for_mock_model() -> None:
    p = get_provider("mock")
    assert isinstance(p, MockLLMProvider)


def test_get_provider_uses_default_when_none() -> None:
    # default_model is reset to "mock-1" by the autouse fixture.
    p = get_provider(None)
    assert isinstance(p, MockLLMProvider)


def test_get_provider_returns_openai_when_model_is_real_and_key_set() -> None:
    settings.openai_api_key = "sk-test"
    settings.openai_base_url = "https://api.example.com/v1"
    p = get_provider("gpt-4o-mini")
    assert isinstance(p, OpenAICompatibleProvider)
    assert p.default_model == "gpt-4o-mini"


def test_get_provider_raises_without_api_key_for_non_mock() -> None:
    assert settings.openai_api_key is None
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_provider("gpt-4o-mini")


def test_provider_constructor_requires_api_key_or_client() -> None:
    with pytest.raises(ValueError, match="openai_api_key is required"):
        OpenAICompatibleProvider(model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_openai_provider_parses_text_response() -> None:
    fake_client = _FakeOpenAIClient(
        completions=[_make_completion(content="Hello, world.", model="gpt-4o-mini")]
    )
    provider = OpenAICompatibleProvider(model="gpt-4o-mini", client=fake_client)
    resp = provider.chat([ChatMessage(role="user", content="hi")])

    assert resp.final_text == "Hello, world."
    assert resp.tool_calls == []
    assert resp.provider == "openai_compatible"
    assert resp.model == "gpt-4o-mini"
    assert resp.token_usage.prompt_tokens == 12
    assert resp.token_usage.completion_tokens == 7
    assert resp.token_usage.total_tokens == 19
    assert resp.estimated_cost_usd is not None
    assert resp.estimated_cost_usd > Decimal("0")
    assert resp.raw_response["finish_reason"] == "stop"


def test_openai_provider_parses_tool_call_response() -> None:
    fake_client = _FakeOpenAIClient(
        completions=[
            _make_completion(
                content=None,
                tool_calls=[
                    {
                        "id": "call_abc",
                        "name": "get_baggage_policy",
                        "arguments": {"cabin_class": "business"},
                    }
                ],
                model="gpt-4o-mini",
            )
        ]
    )
    provider = OpenAICompatibleProvider(model="gpt-4o-mini", client=fake_client)
    resp = provider.chat(
        [ChatMessage(role="user", content="business baggage allowance?")],
        tools=[
            {
                "name": "get_baggage_policy",
                "description": "...",
                "input_schema": {"type": "object", "properties": {"cabin_class": {"type": "string"}}},
            }
        ],
    )

    assert resp.final_text is None
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "get_baggage_policy"
    assert tc.arguments == {"cabin_class": "business"}


def test_openai_provider_handles_malformed_tool_arguments() -> None:
    """Provider must not crash if the model returns non-JSON arguments."""
    bad_tc = SimpleNamespace(
        id="call_x",
        type="function",
        function=SimpleNamespace(name="get_customer_profile", arguments="not-json{"),
    )
    raw = SimpleNamespace(
        id="cmpl-1",
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[bad_tc]),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    fake_client = _FakeOpenAIClient(completions=[raw])
    provider = OpenAICompatibleProvider(model="gpt-4o-mini", client=fake_client)
    resp = provider.chat([ChatMessage(role="user", content="who am i")])
    assert resp.tool_calls[0].arguments == {"_raw_arguments": "not-json{"}


# ---------------------------------------------------------------------------
# Tool-call wire format and fallback
# ---------------------------------------------------------------------------


def test_openai_provider_forwards_tools_in_request() -> None:
    fake_client = _FakeOpenAIClient(completions=[_make_completion(content="hi")])
    provider = OpenAICompatibleProvider(model="gpt-4o-mini", client=fake_client)
    tools = [
        {
            "name": "get_baggage_policy",
            "description": "policy",
            "input_schema": {"type": "object"},
        }
    ]
    provider.chat([ChatMessage(role="user", content="hi")], tools=tools)
    sent_kwargs = fake_client.calls[0]
    assert "tools" in sent_kwargs
    assert sent_kwargs["tools"][0]["type"] == "function"
    assert sent_kwargs["tools"][0]["function"]["name"] == "get_baggage_policy"
    assert sent_kwargs["tools"][0]["function"]["parameters"]["type"] == "object"


def test_openai_provider_falls_back_when_tools_unsupported() -> None:
    """If the first call errors with a tool-related message, retry without tools and flip the flag."""
    fake_client = _FakeOpenAIClient(
        completions=[_make_completion(content="Sure thing.")],
        raise_first_with=RuntimeError(
            "Error: model 'tiny-1' does not support tool calling"
        ),
    )
    provider = OpenAICompatibleProvider(model="tiny-1", client=fake_client)
    resp = provider.chat(
        [ChatMessage(role="user", content="hi")],
        tools=[
            {"name": "x", "description": "x", "input_schema": {"type": "object"}}
        ],
    )
    # Two SDK calls: one with tools (which failed), then a retry without.
    assert len(fake_client.calls) == 2
    assert "tools" in fake_client.calls[0]
    assert "tools" not in fake_client.calls[1]
    assert provider.supports_tools is False
    assert resp.tool_calling_supported is False
    assert resp.final_text == "Sure thing."


def test_openai_provider_propagates_unrelated_errors() -> None:
    fake_client = _FakeOpenAIClient(
        completions=[_make_completion(content="never reached")],
        raise_first_with=RuntimeError("Auth failed: invalid api key"),
    )
    provider = OpenAICompatibleProvider(model="gpt-4o-mini", client=fake_client)
    with pytest.raises(RuntimeError, match="Auth failed"):
        provider.chat(
            [ChatMessage(role="user", content="hi")],
            tools=[{"name": "x", "description": "x", "input_schema": {"type": "object"}}],
        )


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_assistant_tool_call_and_tool_message_serialized_for_sdk() -> None:
    fake_client = _FakeOpenAIClient(completions=[_make_completion(content="ok")])
    provider = OpenAICompatibleProvider(model="gpt-4o-mini", client=fake_client)
    from app.llm import LLMToolCall

    assistant = ChatMessage(
        role="assistant",
        tool_calls=[LLMToolCall(id="c1", name="t1", arguments={"a": 1})],
    )
    tool_msg = ChatMessage(role="tool", tool_call_id="c1", name="t1", content='{"a":1}')
    provider.chat([
        ChatMessage(role="user", content="hi"),
        assistant,
        tool_msg,
    ])
    sent = fake_client.calls[0]["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["tool_calls"][0]["function"]["name"] == "t1"
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"a": 1})
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_call_id"] == "c1"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def test_cost_estimation_known_model() -> None:
    # gpt-4o-mini: $0.15/1M prompt + $0.60/1M completion
    cost = _estimate_cost_usd("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost is not None
    assert cost == Decimal("0.750000")


def test_cost_estimation_unknown_model_returns_none() -> None:
    assert _estimate_cost_usd("acme-unknown-2025", 1000, 1000) is None
