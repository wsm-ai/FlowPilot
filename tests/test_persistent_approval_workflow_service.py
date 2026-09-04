import asyncio

import pytest

from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.repository import PersistenceError
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.providers.base import LLMProviderError
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.persistent_approval_workflow_service import (
    ApprovalRunNotFoundError,
    ApprovalRunNotPendingError,
    ApprovalRunThreadMismatchError,
    PersistentApprovalWorkflowService,
)
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry, create_default_tool_registry


class FakePlannerService:
    def __init__(self, plan=None, error=None) -> None:
        self.plan = plan
        self.error = error

    async def create_plan(self, goal: str):
        if self.error is not None:
            raise self.error
        return self.plan


class SpyIssueTool:
    name = "create_test_issue"
    description = "Create a test issue"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, *, fail: bool = False) -> None:
        self.call_count = 0
        self.fail = fail

    async def execute(self, arguments):
        self.call_count += 1
        if self.fail:
            raise ToolExecutionError("sensitive tool failure")
        return {"issue_id": f"TEST-{self.call_count:03d}", **arguments}


def make_plan(*, approvals: int = 0) -> ExecutionPlan:
    if approvals:
        steps = [
            PlanStep(
                id=index,
                description=f"Create issue {index}",
                action="create_test_issue",
                arguments={"title": f"Issue {index}"},
                requires_approval=True,
            )
            for index in range(1, approvals + 1)
        ]
    else:
        steps = [
            PlanStep(
                id=1,
                description="Read feedback",
                action="get_customer_feedback",
                arguments={"customer_id": "C001", "priority": "high"},
            )
        ]
    return ExecutionPlan(goal="Test plan", steps=steps)


def make_registry(tool: SpyIssueTool | None = None) -> ToolRegistry:
    registry = create_default_tool_registry()
    if tool is not None:
        registry.register(tool)
    return registry


def test_persistent_start_normal_plan_records_completed_run(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            workflow = ApprovalWorkflowService(
                FakePlannerService(make_plan()), make_registry(), saver
            )
            service = PersistentApprovalWorkflowService(workflow, repository)
            result = await service.start("Test plan")
            return result, await repository.get(result.run_id)

    result, record = asyncio.run(scenario())

    assert result.status == "completed"
    assert record.mode == "planned"
    assert record.status == "completed"
    assert record.thread_id == result.thread_id


def test_pending_run_approve_preserves_ids_and_completes(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        tool = SpyIssueTool()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            workflow = ApprovalWorkflowService(
                FakePlannerService(make_plan(approvals=1)),
                make_registry(tool),
                saver,
            )
            service = PersistentApprovalWorkflowService(workflow, repository)
            started = await service.start("Test plan", thread_id="thread-a")
            before = await repository.get(started.run_id)
            resumed = await service.resume(
                started.run_id, started.thread_id, "approve"
            )
            after = await repository.get(started.run_id)
            return started, resumed, before, after, tool

    started, resumed, before, after, tool = asyncio.run(scenario())

    assert started.status == "approval_required"
    assert before.status == "approval_required"
    assert before.thread_id == started.thread_id
    assert tool.call_count == 1
    assert resumed.run_id == started.run_id
    assert resumed.thread_id == started.thread_id
    assert resumed.status == "completed"
    assert after.status == "completed"


def test_reject_is_persisted_as_business_terminal_state(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        tool = SpyIssueTool()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            workflow = ApprovalWorkflowService(
                FakePlannerService(make_plan(approvals=1)),
                make_registry(tool),
                saver,
            )
            service = PersistentApprovalWorkflowService(workflow, repository)
            started = await service.start("Test plan")
            result = await service.resume(
                started.run_id, started.thread_id, "reject"
            )
            return result, await repository.get(started.run_id), tool

    result, record, tool = asyncio.run(scenario())

    assert result.status == "rejected"
    assert record.status == "rejected"
    assert tool.call_count == 0


def test_first_approval_can_persist_second_pending_approval(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        tool = SpyIssueTool()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(
                    FakePlannerService(make_plan(approvals=2)),
                    make_registry(tool),
                    saver,
                ),
                repository,
            )
            started = await service.start("Test plan")
            resumed = await service.resume(
                started.run_id, started.thread_id, "approve"
            )
            return resumed, await repository.get(started.run_id), tool

    result, record, tool = asyncio.run(scenario())

    assert result.status == "approval_required"
    assert result.pending_approval["step_id"] == 2
    assert record.status == "approval_required"
    assert record.result["pending_approval"]["step_id"] == 2
    assert tool.call_count == 1


def test_resume_preflight_rejects_unknown_run_without_graph_resume(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(
                    FakePlannerService(make_plan()), make_registry(), saver
                ),
                repository,
            )
            await service.resume("missing-run", "thread-a", "approve")

    with pytest.raises(ApprovalRunNotFoundError):
        asyncio.run(scenario())


def test_mismatched_thread_does_not_resume_or_execute_tool(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        tool = SpyIssueTool()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(
                    FakePlannerService(make_plan(approvals=1)),
                    make_registry(tool),
                    saver,
                ),
                repository,
            )
            started = await service.start("Test plan", thread_id="correct-thread")
            with pytest.raises(ApprovalRunThreadMismatchError):
                await service.resume(started.run_id, "wrong-thread", "approve")
            return await repository.get(started.run_id), tool

    record, tool = asyncio.run(scenario())

    assert record.status == "approval_required"
    assert tool.call_count == 0


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_terminal_run_cannot_resume_again(tmp_path, decision):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        tool = SpyIssueTool()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(
                    FakePlannerService(make_plan(approvals=1)),
                    make_registry(tool),
                    saver,
                ),
                repository,
            )
            started = await service.start("Test plan")
            await service.resume(started.run_id, started.thread_id, decision)
            calls = tool.call_count
            with pytest.raises(ApprovalRunNotPendingError):
                await service.resume(started.run_id, started.thread_id, decision)
            return calls, tool.call_count

    assert asyncio.run(scenario()) in {(1, 1), (0, 0)}


def test_tool_failure_marks_run_failed_without_advancing_success(tmp_path):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        tool = SpyIssueTool(fail=True)
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(
                    FakePlannerService(make_plan(approvals=1)),
                    make_registry(tool),
                    saver,
                ),
                repository,
            )
            started = await service.start("Test plan")
            with pytest.raises(ToolExecutionError):
                await service.resume(started.run_id, started.thread_id, "approve")
            return await repository.get(started.run_id)

    record = asyncio.run(scenario())

    assert record.status == "failed"
    assert record.error_type == "ToolExecutionError"


@pytest.mark.parametrize(
    "error",
    [LLMProviderError("sensitive llm"), PlanningError("sensitive planner")],
)
def test_start_workflow_failure_marks_run_failed(tmp_path, error):
    async def scenario():
        repository = SQLiteRunRepository(tmp_path / "runs.db")
        await repository.initialize()
        async with async_checkpoint_saver(tmp_path / "checkpoints.db") as saver:
            service = PersistentApprovalWorkflowService(
                ApprovalWorkflowService(
                    FakePlannerService(error=error), make_registry(), saver
                ),
                repository,
            )
            with pytest.raises(type(error)):
                await service.start("Test plan", thread_id="failed-thread")
            return (await repository.list_recent())[0]

    record = asyncio.run(scenario())

    assert record.status == "failed"
    assert record.error_type == type(error).__name__
    assert record.thread_id == "failed-thread"


def test_repository_create_failure_does_not_start_workflow():
    class FailingRepository:
        async def create(self, record):
            raise PersistenceError("create failed")

    class SpyWorkflow:
        call_count = 0

        async def start(self, goal, *, thread_id=None):
            self.call_count += 1

    workflow = SpyWorkflow()
    service = PersistentApprovalWorkflowService(workflow, FailingRepository())

    with pytest.raises(PersistenceError):
        asyncio.run(service.start("Test plan"))

    assert workflow.call_count == 0
