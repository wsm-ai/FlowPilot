import asyncio

import pytest

from app.grounding.models import GroundedAnswer
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.grounded_answer_service import GroundedAnswerError
from app.services.persistent_approval_workflow_service import (
    ApprovalRunNotFoundError,
    ApprovalRunThreadMismatchError,
    GroundedAnswerRetryNotAllowedError,
    PersistentApprovalWorkflowService,
)
from app.tools.registry import create_default_tool_registry


class SpyPlanner:
    def __init__(self) -> None:
        self.call_count = 0

    async def create_plan(self, goal: str) -> ExecutionPlan:
        self.call_count += 1
        return ExecutionPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id=1,
                    description="Create issue",
                    action="create_test_issue",
                    arguments={"title": "Login issue"},
                    requires_approval=True,
                )
            ],
        )


class SpyIssueTool:
    name = "create_test_issue"
    description = "Create a test issue"
    parameters = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, arguments):
        self.call_count += 1
        return {"issue_id": "TEST-1", **arguments}


class SequenceSynthesizer:
    def __init__(self, *, always_fail: bool = False) -> None:
        self.always_fail = always_fail
        self.call_count = 0
        self.calls = []

    async def synthesize(self, *, goal, plan, step_results):
        self.call_count += 1
        self.calls.append((goal, plan, step_results))
        if self.always_fail or self.call_count == 1:
            raise GroundedAnswerError("sensitive synthesis failure")
        return GroundedAnswer(answer="Recovered grounded answer")


async def build_started_service(tmp_path, synthesizer):
    repository = SQLiteRunRepository(tmp_path / "runs.sqlite")
    await repository.initialize()
    planner = SpyPlanner()
    tool = SpyIssueTool()
    registry = create_default_tool_registry()
    registry.register(tool)
    saver_context = async_checkpoint_saver(tmp_path / "checkpoints.sqlite")
    saver = await saver_context.__aenter__()
    workflow = ApprovalWorkflowService(planner, registry, saver, synthesizer)
    service = PersistentApprovalWorkflowService(workflow, repository)
    pending = await service.start("Create issue", thread_id="thread-a")
    failed_answer = await service.resume(pending.run_id, pending.thread_id, "approve")
    return (
        service,
        repository,
        planner,
        tool,
        synthesizer,
        saver_context,
        failed_answer,
    )


def test_retry_success_uses_persisted_snapshot_without_rerunning_workflow(tmp_path):
    async def scenario():
        values = await build_started_service(tmp_path, SequenceSynthesizer())
        service, repository, planner, tool, synthesizer, saver_context, failed = values
        try:
            before = await repository.get(failed.run_id)
            retried = await service.retry_grounded_answer(failed.run_id, failed.thread_id)
            after = await repository.get(failed.run_id)
            return failed, before, retried, after, planner, tool, synthesizer
        finally:
            await saver_context.__aexit__(None, None, None)

    failed, before, retried, after, planner, tool, synthesizer = asyncio.run(scenario())

    assert failed.status == "completed"
    assert failed.grounding_status == "failed"
    assert before.status == "completed"
    assert before.error_type is None
    assert before.result["grounding_status"] == "failed"
    assert before.result["grounded_answer"] is None
    assert before.result["grounding_error_type"] == "GroundedAnswerError"
    assert retried.run_id == failed.run_id
    assert retried.thread_id == failed.thread_id
    assert retried.status == "completed"
    assert retried.grounding_status == "completed"
    assert retried.grounded_answer.answer == "Recovered grounded answer"
    assert after.status == "completed"
    assert after.result["grounded_answer"]["answer"] == "Recovered grounded answer"
    assert after.result["grounding_error_type"] is None
    assert planner.call_count == 1
    assert tool.call_count == 1
    assert synthesizer.call_count == 2
    assert synthesizer.calls[1][0] == "Create issue"
    assert synthesizer.calls[1][1] == ExecutionPlan.model_validate(before.result["plan"])
    assert synthesizer.calls[1][2] == before.result["step_results"]


def test_retry_failure_keeps_completed_run_retryable_without_tool_reexecution(tmp_path):
    async def scenario():
        values = await build_started_service(
            tmp_path, SequenceSynthesizer(always_fail=True)
        )
        service, repository, planner, tool, synthesizer, saver_context, failed = values
        try:
            retried = await service.retry_grounded_answer(failed.run_id, failed.thread_id)
            record = await repository.get(failed.run_id)
            return retried, record, planner, tool, synthesizer
        finally:
            await saver_context.__aexit__(None, None, None)

    retried, record, planner, tool, synthesizer = asyncio.run(scenario())
    assert retried.status == "completed"
    assert retried.grounding_status == "failed"
    assert retried.grounded_answer is None
    assert record.status == "completed"
    assert record.error_type is None
    assert record.result["grounding_status"] == "failed"
    assert tool.call_count == 1
    assert planner.call_count == 1
    assert synthesizer.call_count == 2


def test_retry_preconditions_reject_missing_mismatch_pending_and_completed_answer(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.sqlite")
        await repository.initialize()
        planner = SpyPlanner()
        tool = SpyIssueTool()
        registry = create_default_tool_registry()
        registry.register(tool)
        synthesizer = SequenceSynthesizer()
        async with async_checkpoint_saver(tmp_path / "checkpoints.sqlite") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(planner, registry, saver, synthesizer),
                repository,
            )
            with pytest.raises(ApprovalRunNotFoundError):
                await service.retry_grounded_answer("missing", "thread-a")
            pending = await service.start("Create issue", thread_id="thread-a")
            with pytest.raises(ApprovalRunThreadMismatchError):
                await service.retry_grounded_answer(pending.run_id, "wrong-thread")
            with pytest.raises(GroundedAnswerRetryNotAllowedError):
                await service.retry_grounded_answer(pending.run_id, pending.thread_id)
            rejected_pending = await service.start(
                "Create another issue", thread_id="thread-b"
            )
            rejected = await service.resume(
                rejected_pending.run_id,
                rejected_pending.thread_id,
                "reject",
            )
            with pytest.raises(GroundedAnswerRetryNotAllowedError):
                await service.retry_grounded_answer(
                    rejected.run_id, rejected.thread_id
                )
            failed = await service.resume(pending.run_id, pending.thread_id, "approve")
            completed = await service.retry_grounded_answer(failed.run_id, failed.thread_id)
            with pytest.raises(GroundedAnswerRetryNotAllowedError):
                await service.retry_grounded_answer(completed.run_id, completed.thread_id)

    asyncio.run(scenario())
