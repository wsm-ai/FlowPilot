import json
from typing import Any, Awaitable, Callable, Literal

from app.graph.state import AgentState
from app.providers.base import LLMProviderError
from app.services.llm_service import LLMService
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


def create_llm_node(
    llm_service: LLMService,
    tools: list[dict[str, Any]],
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def llm_node(state: AgentState) -> dict[str, Any]:
        response = await llm_service.complete(
            messages=state["messages"],
            tools=tools,
            tool_choice="auto",
        )
        update: dict[str, Any] = {
            "llm_response": response,
            "answer": None if response.tool_calls else response.content,
        }
        if not response.tool_calls and response.content:
            update["messages"] = [
                {
                    "role": "assistant",
                    "content": response.content,
                }
            ]
        return update

    return llm_node


def route_after_llm(state: AgentState) -> Literal["tools", "end"]:
    response = state["llm_response"]
    if response is not None and response.tool_calls:
        return "tools"
    return "end"


def create_tool_node(
    registry: ToolRegistry,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def tool_node(state: AgentState) -> dict[str, Any]:
        response = state["llm_response"]
        if response is None:
            raise ToolExecutionError("Cannot execute tools without an LLM response")

        new_messages: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in response.tool_calls
                ],
            }
        ]

        executed_tools = list(state["executed_tools"])
        for tool_call in response.tool_calls:
            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError as exc:
                raise ToolExecutionError("Tool arguments are not valid JSON") from exc

            if not isinstance(arguments, dict):
                raise ToolExecutionError("Tool arguments must be a JSON object")

            result = await registry.execute(tool_call.name, arguments)
            try:
                serialized_result = json.dumps(result, ensure_ascii=False)
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    "Tool result is not JSON serializable"
                ) from exc

            executed_tools.append(
                {
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": arguments,
                }
            )
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": serialized_result,
                }
            )

        return {
            "messages": new_messages,
            "executed_tools": executed_tools,
        }

    return tool_node


def create_final_llm_node(
    llm_service: LLMService,
    tools: list[dict[str, Any]],
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    async def final_llm_node(state: AgentState) -> dict[str, Any]:
        final_response = await llm_service.complete(
            messages=state["messages"],
            tools=tools,
            tool_choice="none",
        )
        if not final_response.content:
            raise LLMProviderError("The language model returned no final answer")

        return {
            "llm_response": final_response,
            "answer": final_response.content,
            "messages": [
                {
                    "role": "assistant",
                    "content": final_response.content,
                }
            ],
        }

    return final_llm_node
