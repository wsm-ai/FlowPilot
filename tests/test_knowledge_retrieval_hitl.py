import asyncio
import json

from app.persistence.checkpoint import async_checkpoint_saver
from app.providers.types import LLMResponse
from app.retrieval.models import KnowledgeChunk, RetrievalQuery, RetrievalResult
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import create_default_tool_registry


class FakeRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.call_count += 1
        return [
            RetrievalResult(
                chunk=KnowledgeChunk(
                    chunk_id="KB-HITL:chunk:0", document_id="KB-HITL",
                    content="Internal evidence", source="handbook", metadata={}, position=0
                ),
                score=1.0, rank=1,
            )
        ]


class PlanProvider:
    model = "test-model"

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        return LLMResponse(content=json.dumps({
            "goal": "Search knowledge",
            "steps": [{
                "id": 1, "description": "Search internal knowledge",
                "action": "search_knowledge_base",
                "arguments": {"query": "Enterprise login troubleshooting"},
                "requires_approval": True,
            }],
        }))


def test_read_only_retrieval_policy_completes_without_interrupt(tmp_path):
    async def scenario():
        retriever = FakeRetriever()
        registry = create_default_tool_registry()
        registry.register(KnowledgeBaseTool(retriever))
        planner = PlannerService(
            LLMService(PlanProvider()), tool_definitions=registry.definitions()
        )
        async with async_checkpoint_saver(tmp_path / "retrieval.sqlite") as saver:
            service = ApprovalWorkflowService(planner, registry, saver)
            result = await service.start("Search knowledge", thread_id="kb-thread")
            snapshot = await service._graph.aget_state(
                {"configurable": {"thread_id": "kb-thread"}}
            )
        return result, snapshot, retriever

    result, snapshot, retriever = asyncio.run(scenario())

    assert result.plan.steps[0].requires_approval is False
    assert result.status == "completed"
    assert result.pending_approval is None
    assert result.current_step_index == 1
    assert result.step_results[0]["action"] == "search_knowledge_base"
    assert result.step_results[0]["result"][0]["chunk_id"] == "KB-HITL:chunk:0"
    assert retriever.call_count == 1
    assert snapshot.next == ()
