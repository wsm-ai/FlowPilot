from typing import Any

from app.providers.base import LLMProvider
from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse


class LLMService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def model(self) -> str:
        return self._provider.model

    async def chat(self, message: str) -> str:
        response = await self._provider.complete(
            messages=[{"role": "user", "content": message}]
        )
        if not response.content:
            raise LLMProviderError(
                "The language model did not return content for a chat request"
            )
        return response.content

    async def choose_tools(
        self,
        message: str,
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        return await self._provider.complete(
            messages=[{"role": "user", "content": message}],
            tools=tools,
            tool_choice="auto",
        )
