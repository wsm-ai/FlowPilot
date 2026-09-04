from typing import Any, Callable, Awaitable

from app.graph.state import AgentState
from app.services.llm_service import LLMService


def create_llm_node(
    llm_service: LLMService,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def llm_node(state: AgentState) -> dict[str, Any]:
        response = await llm_service.complete(messages=state["messages"])
        return {
            "llm_response": response,
            "answer": response.content,
        }

    return llm_node
