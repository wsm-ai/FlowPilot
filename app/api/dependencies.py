from fastapi import Depends

from app.core.config import Settings, get_settings
from app.providers.deepseek import DeepSeekProvider
from app.services.llm_service import LLMService


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    return LLMService(DeepSeekProvider(settings))
