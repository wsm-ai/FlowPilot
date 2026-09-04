import json
from dataclasses import dataclass, field
from typing import Any

from app.providers.base import LLMProviderError
from app.providers.types import LLMToolCall
from app.services.llm_service import LLMService
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


@dataclass(slots=True)
class ExecutedTool:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolCallingResult:
    answer: str
    executed_tools: list[ExecutedTool] = field(default_factory=list)


class ToolCallingService:
    def __init__(self, llm_service: LLMService, registry: ToolRegistry) -> None:
        self._llm_service = llm_service
        self._registry = registry

    async def run(self, message: str) -> ToolCallingResult:
        tools = self._registry.definitions()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": message}
        ]
        first_response = await self._llm_service.complete(
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        if not first_response.tool_calls:
            if not first_response.content:
                raise LLMProviderError("The language model returned no answer")
            return ToolCallingResult(answer=first_response.content)

        messages.append(
            {
                "role": "assistant",
                "content": first_response.content,
                "tool_calls": [
                    self._tool_call_message(tool_call)
                    for tool_call in first_response.tool_calls
                ],
            }
        )

        executed_tools: list[ExecutedTool] = []
        for tool_call in first_response.tool_calls:
            arguments = self._parse_arguments(tool_call.arguments)
            result = await self._registry.execute(tool_call.name, arguments)
            serialized_result = self._serialize_result(result)
            executed_tools.append(
                ExecutedTool(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    arguments=arguments,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": serialized_result,
                }
            )

        final_response = await self._llm_service.complete(
            messages=messages,
            tools=tools,
            tool_choice="none",
        )
        if not final_response.content:
            raise LLMProviderError("The language model returned no final answer")

        return ToolCallingResult(
            answer=final_response.content,
            executed_tools=executed_tools,
        )

    @staticmethod
    def _parse_arguments(raw_arguments: str) -> dict[str, Any]:
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError("Tool arguments are not valid JSON") from exc

        if not isinstance(arguments, dict):
            raise ToolExecutionError("Tool arguments must be a JSON object")
        return arguments

    @staticmethod
    def _serialize_result(result: Any) -> str:
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("Tool result is not JSON serializable") from exc

    @staticmethod
    def _tool_call_message(tool_call: LLMToolCall) -> dict[str, Any]:
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
        }
