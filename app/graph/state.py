from typing import Any, NotRequired, TypedDict

from app.providers.types import LLMResponse
from app.schemas.planning import ExecutionPlan


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    llm_response: LLMResponse | None
    answer: str | None
    executed_tools: list[dict[str, Any]]
    goal: NotRequired[str | None]
    plan: NotRequired[ExecutionPlan | None]
