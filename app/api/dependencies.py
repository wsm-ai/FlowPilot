from fastapi import Depends, Request
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.config import Settings, get_settings
from app.persistence.repository import RunRepository
from app.providers.deepseek import DeepSeekProvider
from app.services.llm_service import LLMService


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    return LLMService(DeepSeekProvider(settings))


def get_run_repository(request: Request) -> RunRepository:
    return request.app.state.run_repository


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    return request.app.state.checkpointer
