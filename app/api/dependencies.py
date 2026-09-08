from fastapi import Depends, Request
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.config import Settings, get_settings
from app.persistence.repository import RunRepository
from app.reliability.side_effects import SideEffectExecutor
from app.providers.deepseek import DeepSeekProvider
from app.retrieval.base import Retriever
from app.services.llm_service import LLMService
from app.tools.registry import ToolRegistry


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    return LLMService(DeepSeekProvider(settings))


def get_run_repository(request: Request) -> RunRepository:
    return request.app.state.run_repository


def get_side_effect_executor(request: Request) -> SideEffectExecutor:
    return request.app.state.side_effect_executor


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    return request.app.state.checkpointer


def get_retriever(request: Request) -> Retriever:
    try:
        return request.app.state.retriever
    except AttributeError as exc:
        raise RuntimeError(
            "Knowledge base retriever is not initialized"
        ) from exc


def get_tool_registry(
    request: Request,
) -> ToolRegistry:
    try:
        return request.app.state.tool_registry
    except AttributeError as exc:
        raise RuntimeError("Tool registry is not initialized") from exc


def get_mcp_approval_required_actions(request: Request) -> frozenset[str]:
    return request.app.state.mcp_approval_required_actions


def get_reactive_tool_registry(request: Request) -> ToolRegistry:
    registry = get_tool_registry(request)
    approval_required_actions = get_mcp_approval_required_actions(request)
    return registry.excluding(approval_required_actions)
