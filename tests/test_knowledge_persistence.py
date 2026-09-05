import asyncio
import json

from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.retrieval.models import KnowledgeChunk, RetrievalQuery, RetrievalResult
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.persistent_approval_workflow_service import PersistentApprovalWorkflowService
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import create_default_tool_registry


class FakePlanner:
    async def create_plan(self, goal: str) -> ExecutionPlan:
        return ExecutionPlan(
            goal=goal,
            steps=[PlanStep(
                id=1, description="Search internal knowledge",
                action="search_knowledge_base",
                arguments={"query": "Enterprise login troubleshooting"},
                requires_approval=False,
            )],
        )


class FakeRetriever:
    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return [RetrievalResult(
            chunk=KnowledgeChunk(
                chunk_id="KB-PERSIST:chunk:0", document_id="KB-PERSIST",
                content="Persisted internal evidence", source="support-guide",
                metadata={"department": "support"}, position=0,
            ),
            score=0.88, rank=1,
        )]


def test_retrieval_evidence_is_persisted_as_json_safe_run_result(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.sqlite")
        await repository.initialize()
        registry = create_default_tool_registry()
        registry.register(KnowledgeBaseTool(FakeRetriever()))
        async with async_checkpoint_saver(tmp_path / "checkpoints.sqlite") as saver:
            workflow = ApprovalWorkflowService(FakePlanner(), registry, saver)
            service = PersistentApprovalWorkflowService(workflow, repository)
            result = await service.start("Search knowledge", thread_id="persist-thread")
            record = await repository.get(result.run_id)
            snapshot = await workflow._graph.aget_state(
                {"configurable": {"thread_id": result.thread_id}}
            )
        return result, record, snapshot

    result, record, snapshot = asyncio.run(scenario())

    assert result.status == "completed"
    assert record is not None
    assert record.status == "completed"
    assert record.thread_id == result.thread_id
    step = record.result["step_results"][0]
    assert step["action"] == "search_knowledge_base"
    evidence = step["result"][0]
    assert evidence["chunk_id"] == "KB-PERSIST:chunk:0"
    assert evidence["document_id"] == "KB-PERSIST"
    assert evidence["source"] == "support-guide"
    assert evidence["content"] == "Persisted internal evidence"
    json.dumps(record.result, allow_nan=False)
    assert snapshot.values["step_results"][0]["result"][0] == evidence
