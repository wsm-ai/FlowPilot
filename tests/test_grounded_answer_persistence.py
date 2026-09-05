import asyncio
import json

from app.grounding.models import Citation, GroundedAnswer
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.providers.types import LLMResponse
from app.retrieval.models import KnowledgeChunk, RetrievalQuery, RetrievalResult
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.grounded_answer_service import GroundedAnswerService
from app.services.llm_service import LLMService
from app.services.persistent_approval_workflow_service import PersistentApprovalWorkflowService
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import create_default_tool_registry


class FakePlanner:
    async def create_plan(self, goal: str) -> ExecutionPlan:
        return ExecutionPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id=1,
                    description="Search internal knowledge",
                    action="search_knowledge_base",
                    arguments={"query": "enterprise login"},
                    requires_approval=False,
                )
            ],
        )


class FakeRetriever:
    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk=KnowledgeChunk(
                    chunk_id="CH-1",
                    document_id="DOC-1",
                    content="Check enterprise SSO configuration.",
                    source="support-handbook",
                    metadata={"document_title": "Support Guide"},
                    position=0,
                ),
                score=0.99,
                rank=1,
            )
        ]


class SynthesisProvider:
    model = "test-model"

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "answer": "Check enterprise SSO configuration.",
                    "support_basis": "knowledge_evidence",
                    "citation_ids": ["E1"],
                }
            )
        )


class ApprovalPlanner:
    async def create_plan(self, goal: str) -> ExecutionPlan:
        return ExecutionPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id=1,
                    description="Create issue",
                    action="create_test_issue",
                    arguments={},
                    requires_approval=True,
                )
            ],
        )


class IssueTool:
    name = "create_test_issue"
    description = "Create a test issue"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return {"issue_id": "TEST-1"}


class FakeSynthesis:
    async def synthesize(self, *, goal, plan, step_results):
        return GroundedAnswer(
            answer="Approved answer",
            citations=[
                Citation(
                    citation_id="E1",
                    chunk_id="CH-1",
                    document_id="DOC-1",
                    source="support-handbook",
                    title="Support Guide",
                )
            ],
        )


def test_real_grounded_answer_round_trips_through_json_persistence(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.sqlite")
        await repository.initialize()
        registry = create_default_tool_registry()
        registry.register(KnowledgeBaseTool(FakeRetriever()))
        async with async_checkpoint_saver(tmp_path / "checkpoints.sqlite") as saver:
            workflow = ApprovalWorkflowService(
                FakePlanner(),
                registry,
                saver,
                GroundedAnswerService(LLMService(SynthesisProvider())),
            )
            service = PersistentApprovalWorkflowService(workflow, repository)
            result = await service.start("Find enterprise login guidance")
            record = await repository.get(result.run_id)
        return result, record

    result, record = asyncio.run(scenario())

    assert result.status == "completed"
    assert result.grounding_status == "completed"
    assert result.grounded_answer.answer == "Check enterprise SSO configuration."
    assert result.grounded_answer.citations[0].chunk_id == "CH-1"
    persisted = record.result["grounded_answer"]
    assert ExecutionPlan.model_validate(record.result["plan"]) == result.plan
    assert record.result["grounding_status"] == "completed"
    assert record.result["grounding_error_type"] is None
    assert persisted == result.grounded_answer.model_dump(mode="json")
    assert persisted["citations"][0] == {
        "citation_id": "E1",
        "chunk_id": "CH-1",
        "document_id": "DOC-1",
        "source": "support-handbook",
        "title": "Support Guide",
    }
    json.dumps(record.result, allow_nan=False)


def test_pending_and_approved_resume_update_same_run(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.sqlite")
        await repository.initialize()
        registry = create_default_tool_registry()
        registry.register(IssueTool())
        async with async_checkpoint_saver(tmp_path / "checkpoints.sqlite") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(
                    ApprovalPlanner(), registry, saver, FakeSynthesis()
                ),
                repository,
            )
            pending = await service.start("Create issue", thread_id="thread-a")
            pending_record = await repository.get(pending.run_id)
            completed = await service.resume(
                pending.run_id, pending.thread_id, "approve"
            )
            completed_record = await repository.get(pending.run_id)
        return pending, pending_record, completed, completed_record

    pending, pending_record, completed, completed_record = asyncio.run(scenario())

    assert pending.grounded_answer is None
    assert pending.grounding_status == "not_attempted"
    assert pending_record.result["grounded_answer"] is None
    assert pending_record.result["grounding_status"] == "not_attempted"
    assert completed.run_id == pending.run_id
    assert completed.status == "completed"
    assert completed.grounding_status == "completed"
    assert completed.grounded_answer.answer == "Approved answer"
    assert completed_record.result["grounded_answer"]["answer"] == "Approved answer"
