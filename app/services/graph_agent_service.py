from dataclasses import dataclass, field
from typing import Any

from app.graph.workflow import create_basic_agent_graph
from app.providers.base import LLMProviderError
from app.services.llm_service import LLMService
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


@dataclass(slots=True)
class GraphExecutedTool:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class GraphAgentResult:
    answer: str
    executed_tools: list[GraphExecutedTool] = field(default_factory=list)


class GraphAgentService:
    def __init__(self, llm_service: LLMService, registry: ToolRegistry) -> None:
        self._graph = create_basic_agent_graph(llm_service, registry)

    async def run(self, message: str) -> GraphAgentResult:
        result = await self._graph.ainvoke(
            {
                "messages": [{"role": "user", "content": message}],
                "llm_response": None,
                "answer": None,
                "executed_tools": [],
            }
        )

        answer = result.get("answer")
        if not answer:
            raise LLMProviderError("The language model returned no final answer")

        return GraphAgentResult(
            answer=answer,
            executed_tools=self._convert_executed_tools(
                result.get("executed_tools", [])
            ),
        )

    @staticmethod
    def _convert_executed_tools(
        records: Any,
    ) -> list[GraphExecutedTool]:
        if not isinstance(records, list):
            raise ToolExecutionError("Invalid executed tools state")

        converted: list[GraphExecutedTool] = []
        for record in records:
            if not isinstance(record, dict):
                raise ToolExecutionError("Invalid executed tool record")

            tool_call_id = record.get("tool_call_id")
            name = record.get("name")
            arguments = record.get("arguments")
            if (
                not isinstance(tool_call_id, str)
                or not isinstance(name, str)
                or not isinstance(arguments, dict)
            ):
                raise ToolExecutionError("Invalid executed tool record")

            converted.append(
                GraphExecutedTool(
                    tool_call_id=tool_call_id,
                    name=name,
                    arguments=arguments.copy(),
                )
            )
        return converted
