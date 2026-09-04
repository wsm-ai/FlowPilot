from dataclasses import dataclass, field

from app.persistence.models import create_agent_run_record
from app.persistence.repository import RunRepository
from app.providers.base import LLMProviderError
from app.services.graph_agent_service import GraphAgentService, GraphExecutedTool
from app.tools.base import ToolExecutionError


@dataclass(slots=True)
class PersistentAgentResult:
    run_id: str
    answer: str
    executed_tools: list[GraphExecutedTool] = field(default_factory=list)


class PersistentAgentService:
    def __init__(
        self,
        graph_agent_service: GraphAgentService,
        run_repository: RunRepository,
    ) -> None:
        self._graph_agent_service = graph_agent_service
        self._run_repository = run_repository

    async def run(
        self,
        message: str,
        *,
        thread_id: str | None = None,
    ) -> PersistentAgentResult:
        record = create_agent_run_record(mode="reactive", input_text=message)
        await self._run_repository.create(record)

        try:
            result = await self._graph_agent_service.run(
                message,
                thread_id=thread_id,
            )
        except (LLMProviderError, ToolExecutionError) as exc:
            await self._run_repository.update(
                record.run_id,
                status="failed",
                error_type=type(exc).__name__,
            )
            raise

        safe_result = {
            "answer": result.answer,
            "executed_tools": [
                {
                    "tool_call_id": tool.tool_call_id,
                    "name": tool.name,
                    "arguments": tool.arguments,
                }
                for tool in result.executed_tools
            ],
        }
        await self._run_repository.update(
            record.run_id,
            status="completed",
            result=safe_result,
        )
        return PersistentAgentResult(
            run_id=record.run_id,
            answer=result.answer,
            executed_tools=result.executed_tools,
        )
