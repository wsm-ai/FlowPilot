from typing import Any, TypedDict

from app.providers.types import LLMResponse


class AgentState(TypedDict):
    messages: list[dict[str, Any]]
    llm_response: LLMResponse | None
    answer: str | None
    executed_tools: list[dict[str, Any]]
