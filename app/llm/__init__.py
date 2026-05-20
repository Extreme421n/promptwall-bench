"""LLM provider abstraction + concrete providers.

The chatbot talks to providers exclusively through ``LLMProvider``. New
providers are added by implementing ``chat`` and being wired into
``get_provider``.
"""

from typing import Optional

from app.config import settings
from app.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    TokenUsage,
    messages_to_dicts,
)
from app.llm.mock import MockLLMProvider
from app.llm.openai_compatible import OpenAICompatibleProvider


def get_provider(model: Optional[str] = None) -> LLMProvider:
    """Resolve a model id to a provider instance.

    Routing rules:

    - ``model`` starting with ``mock`` (or empty + default_model="mock-*")
      → :class:`MockLLMProvider`.
    - Anything else → :class:`OpenAICompatibleProvider`, configured from env
      (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``).

    Raises ``ValueError`` when a non-mock model is requested without an API
    key configured, so callers can return a clean 400.
    """
    resolved = (model or settings.default_model or "mock-1").strip()
    if resolved.startswith("mock"):
        return MockLLMProvider(model=resolved)
    if not settings.openai_api_key:
        raise ValueError(
            f"OPENAI_API_KEY is not configured; cannot serve model {resolved!r}"
        )
    return OpenAICompatibleProvider(
        model=resolved,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


__all__ = [
    "ChatMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMToolCall",
    "TokenUsage",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "get_provider",
    "messages_to_dicts",
]
