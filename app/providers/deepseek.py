from typing import Any

from openai import AsyncOpenAI, OpenAIError

from app.core.config import Settings
from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse, LLMToolCall


class DeepSeekProvider:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.deepseek_model
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            timeout=30.0,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            request.update(
                tools=tools,
                tool_choice=tool_choice or "auto",
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            response = await self._client.chat.completions.create(**request)
        except OpenAIError as exc:
            raise LLMProviderError("The language model service is unavailable") from exc

        if not response.choices:
            raise LLMProviderError("The language model returned no choices")

        message = response.choices[0].message
        content = message.content or None
        tool_calls = [
            LLMToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            )
            for tool_call in (message.tool_calls or [])
        ]

        if content is None and not tool_calls:
            raise LLMProviderError("The language model returned an empty response")

        return LLMResponse(content=content, tool_calls=tool_calls)
