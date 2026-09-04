from typing import Annotated, Any, NotRequired, TypedDict

from app.graph.reducers import add_messages
from app.providers.types import LLMResponse
from app.schemas.planning import ExecutionPlan


class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], add_messages]
    llm_response: LLMResponse | None
    answer: str | None
    executed_tools: list[dict[str, Any]]
    goal: NotRequired[str | None]
    plan: NotRequired[ExecutionPlan | None]
    current_step_index: NotRequired[int]
    route: NotRequired[str | None]
    step_results: NotRequired[list[dict[str, Any]]]
    pending_approval: NotRequired[dict[str, Any] | None]
    approval_decision: NotRequired[str | None]
