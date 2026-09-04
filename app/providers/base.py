from typing import Any, Protocol

from app.providers.types import LLMResponse


class LLMProviderError(Exception):
    """Raised when an LLM provider cannot complete a request."""


class LLMProvider(Protocol):
    """Minimal interface implemented by language model providers."""

    model: str

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        ...
