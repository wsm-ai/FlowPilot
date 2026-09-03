from openai import AsyncOpenAI, OpenAIError

from app.core.config import Settings
from app.providers.base import LLMProviderError


class DeepSeekProvider:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.deepseek_model
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            timeout=30.0,
        )

    async def chat(self, messages: list[dict[str, str]]) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except OpenAIError as exc:
            raise LLMProviderError("The language model service is unavailable") from exc

        if not response.choices:
            raise LLMProviderError("The language model returned no choices")

        content = response.choices[0].message.content
        if not content:
            raise LLMProviderError("The language model returned an empty response")
        return content
