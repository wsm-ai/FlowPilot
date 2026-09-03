from app.providers.base import LLMProvider


class LLMService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def model(self) -> str:
        return self._provider.model

    async def chat(self, message: str) -> str:
        return await self._provider.chat(
            messages=[{"role": "user", "content": message}]
        )
