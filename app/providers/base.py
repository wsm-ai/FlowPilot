from typing import Protocol


class LLMProviderError(Exception):
    """Raised when an LLM provider cannot complete a request."""


class LLMProvider(Protocol):
    """Minimal interface implemented by language model providers."""

    model: str

    async def chat(self, messages: list[dict[str, str]]) -> str:
        ...
