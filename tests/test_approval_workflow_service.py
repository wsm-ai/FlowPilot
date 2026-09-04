import asyncio

import pytest

from app.persistence.checkpoint import async_checkpoint_saver
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import (
    ApprovalNotPendingError,
    ApprovalThreadConflictError,
    ApprovalThreadNotFoundError,
    ApprovalWorkflowService,
)
from app.tools.registry import ToolRegistry, create_default_tool_registry


class FakePlannerService:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.call_count = 0

    async def create_plan(self, goal: str) -> ExecutionPlan:
        self.call_count += 1
        return self.plan


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
        return {"issue_id": f"TEST-{self.call_count:03d}", **arguments}


def normal_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Review feedback",
        steps=[
            PlanStep(
                id=1,
                description="Read feedback",
                action="get_customer_feedback",
                arguments={"customer_id": "C001", "priority": "high"},
            )
        ],
    )


def approval_plan(step_count: int = 1) -> ExecutionPlan:
    return ExecutionPlan(
        goal="Create test issues",
        steps=[
            PlanStep(
                id=index,
                description=f"Create test issue {index}",
                action="create_test_issue",
                arguments={"title": f"Issue {index}"},
                requires_approval=True,
            )
            for index in range(1, step_count + 1)
        ],
    )


def registry_with_spy() -> tuple[ToolRegistry, SpyIssueTool]:
    registry = create_default_tool_registry()
    tool = SpyIssueTool()
    registry.register(tool)
    return registry, tool


def test_start_normal_plan_completes(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "normal.sqlite") as saver:
            service = ApprovalWorkflowService(
                FakePlannerService(normal_plan()),
                create_default_tool_registry(),
                saver,
            )
            return await service.start("Review feedback")

    result = asyncio.run(scenario())

    assert result.status == "completed"
    assert result.current_step_index == 1
    assert len(result.step_results) == 1
    assert result.pending_approval is None
    assert result.thread_id


def test_start_approval_plan_returns_pending_payload_without_execution(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "pending.sqlite") as saver:
            registry, tool = registry_with_spy()
            service = ApprovalWorkflowService(
                FakePlannerService(approval_plan()), registry, saver
            )
            result = await service.start("Create issue", thread_id="client-thread")
            return result, tool

    result, tool = asyncio.run(scenario())

    assert result.thread_id == "client-thread"
    assert result.status == "approval_required"
    assert result.pending_approval["step_id"] == 1
    assert result.pending_approval["action"] == "create_test_issue"
    assert tool.call_count == 0


def test_start_generates_thread_id_when_omitted(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "uuid.sqlite") as saver:
            service = ApprovalWorkflowService(
                FakePlannerService(normal_plan()),
                create_default_tool_registry(),
                saver,
            )
            return await service.start("Review feedback")

    assert asyncio.run(scenario()).thread_id


def test_start_rejects_existing_thread_id(tmp_path):
    planner = FakePlannerService(normal_plan())

    async def scenario():
        async with async_checkpoint_saver(tmp_path / "conflict.sqlite") as saver:
            service = ApprovalWorkflowService(
                planner,
                create_default_tool_registry(),
                saver,
            )
            await service.start("First plan", thread_id="same-thread")
            await service.start("Second plan", thread_id="same-thread")

    with pytest.raises(ApprovalThreadConflictError):
        asyncio.run(scenario())

    assert planner.call_count == 1


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_calls", "expected_index"),
    [
        ("approve", "completed", 1, 1),
        ("reject", "rejected", 0, 0),
    ],
)
def test_resume_approve_or_reject(
    tmp_path,
    decision,
    expected_status,
    expected_calls,
    expected_index,
):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / f"{decision}.sqlite") as saver:
            registry, tool = registry_with_spy()
            service = ApprovalWorkflowService(
                FakePlannerService(approval_plan()), registry, saver
            )
            await service.start("Create issue", thread_id="approval-thread")
            result = await service.resume("approval-thread", decision)
            return result, tool

    result, tool = asyncio.run(scenario())

    assert result.status == expected_status
    assert result.current_step_index == expected_index
    assert result.pending_approval is None
    assert tool.call_count == expected_calls


def test_resume_rejects_unknown_thread(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "missing.sqlite") as saver:
            service = ApprovalWorkflowService(
                FakePlannerService(normal_plan()),
                create_default_tool_registry(),
                saver,
            )
            await service.resume("missing-thread", "approve")

    with pytest.raises(ApprovalThreadNotFoundError):
        asyncio.run(scenario())


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_finished_thread_cannot_be_resumed_again(tmp_path, decision):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / f"finished-{decision}.sqlite") as saver:
            registry, tool = registry_with_spy()
            service = ApprovalWorkflowService(
                FakePlannerService(approval_plan()), registry, saver
            )
            await service.start("Create issue", thread_id="finished-thread")
            await service.resume("finished-thread", decision)
            calls_before_retry = tool.call_count
            with pytest.raises(ApprovalNotPendingError):
                await service.resume("finished-thread", decision)
            return calls_before_retry, tool.call_count

    calls_before_retry, calls_after_retry = asyncio.run(scenario())

    assert calls_after_retry == calls_before_retry


def test_approve_can_advance_to_another_pending_approval(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "multiple.sqlite") as saver:
            registry, tool = registry_with_spy()
            service = ApprovalWorkflowService(
                FakePlannerService(approval_plan(2)), registry, saver
            )
            await service.start("Create issues", thread_id="multiple-thread")
            result = await service.resume("multiple-thread", "approve")
            return result, tool

    result, tool = asyncio.run(scenario())

    assert result.status == "approval_required"
    assert result.current_step_index == 1
    assert result.pending_approval["step_id"] == 2
    assert len(result.step_results) == 1
    assert tool.call_count == 1


def test_resume_survives_sqlite_saver_reconnection(tmp_path):
    database_path = tmp_path / "durable-service.sqlite"
    registry, tool = registry_with_spy()

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver_a:
            service_a = ApprovalWorkflowService(
                FakePlannerService(approval_plan()), registry, saver_a
            )
            started = await service_a.start(
                "Create issue", thread_id="durable-thread"
            )

        async with async_checkpoint_saver(database_path) as saver_b:
            service_b = ApprovalWorkflowService(
                FakePlannerService(approval_plan()), registry, saver_b
            )
            resumed = await service_b.resume("durable-thread", "approve")
            return started, resumed

    started, resumed = asyncio.run(scenario())

    assert started.status == "approval_required"
    assert resumed.status == "completed"
    assert resumed.thread_id == started.thread_id
    assert tool.call_count == 1
