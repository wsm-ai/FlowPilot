import asyncio

import pytest

from app.grounding.models import GroundedAnswer
from app.persistence.checkpoint import async_checkpoint_saver
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.grounded_answer_service import GroundedAnswerError
from app.tools.registry import ToolRegistry, create_default_tool_registry


class FakePlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan

    async def create_plan(self, goal: str) -> ExecutionPlan:
        return self.plan


class SpyTool:
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


class FakeGroundedAnswerService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    async def synthesize(self, *, goal, plan, step_results):
        self.calls.append(
            {"goal": goal, "plan": plan, "step_results": step_results}
        )
        if self.error is not None:
            raise self.error
        return GroundedAnswer(answer="Grounded final answer")


def feedback_plan(goal="Review customer feedback"):
    return ExecutionPlan(
        goal=goal,
        steps=[
            PlanStep(
                id=1,
                description="Read customer feedback",
                action="get_customer_feedback",
                arguments={"customer_id": "C001", "priority": "high"},
                requires_approval=False,
            )
        ],
    )


def approval_plan(goal="Create an issue"):
    return ExecutionPlan(
        goal=goal,
        steps=[
            PlanStep(
                id=1,
                description="Create an issue",
                action="create_test_issue",
                arguments={"title": "Login failure"},
                requires_approval=True,
            )
        ],
    )


def registry_with_spy() -> tuple[ToolRegistry, SpyTool]:
    registry = create_default_tool_registry()
    tool = SpyTool()
    registry.register(tool)
    return registry, tool


def test_immediately_completed_workflow_synthesizes_once(tmp_path):
    async def scenario():
        synthesis = FakeGroundedAnswerService()
        async with async_checkpoint_saver(tmp_path / "completed.sqlite") as saver:
            service = ApprovalWorkflowService(
                FakePlanner(feedback_plan()),
                create_default_tool_registry(),
                saver,
                synthesis,
            )
            result = await service.start("Original workflow goal")
        return result, synthesis

    result, synthesis = asyncio.run(scenario())
    assert result.status == "completed"
    assert result.grounded_answer.answer == "Grounded final answer"
    assert len(synthesis.calls) == 1
    assert synthesis.calls[0]["goal"] == "Original workflow goal"
    assert synthesis.calls[0]["step_results"] == result.step_results


@pytest.mark.parametrize("decision", [None, "reject"])
def test_pending_or_rejected_workflow_does_not_synthesize(tmp_path, decision):
    async def scenario():
        registry, tool = registry_with_spy()
        synthesis = FakeGroundedAnswerService()
        async with async_checkpoint_saver(tmp_path / f"{decision}.sqlite") as saver:
            service = ApprovalWorkflowService(
                FakePlanner(approval_plan()), registry, saver, synthesis
            )
            result = await service.start("Create an issue", thread_id="approval-thread")
            if decision is not None:
                result = await service.resume("approval-thread", decision)
        return result, synthesis, tool

    result, synthesis, tool = asyncio.run(scenario())
    assert result.status == ("approval_required" if decision is None else "rejected")
    assert result.grounded_answer is None
    assert synthesis.calls == []
    assert tool.call_count == 0


def test_final_approval_completion_synthesizes_once_using_resumed_state_goal(tmp_path):
    async def scenario():
        registry, tool = registry_with_spy()
        synthesis = FakeGroundedAnswerService()
        async with async_checkpoint_saver(tmp_path / "approved.sqlite") as saver:
            service = ApprovalWorkflowService(
                FakePlanner(approval_plan()), registry, saver, synthesis
            )
            started = await service.start("Original approval goal", thread_id="thread-a")
            resumed = await service.resume("thread-a", "approve")
        return started, resumed, synthesis, tool

    started, resumed, synthesis, tool = asyncio.run(scenario())
    assert started.status == "approval_required"
    assert started.grounded_answer is None
    assert resumed.status == "completed"
    assert resumed.grounded_answer.answer == "Grounded final answer"
    assert len(synthesis.calls) == 1
    assert synthesis.calls[0]["goal"] == "Original approval goal"
    assert synthesis.calls[0]["step_results"] == resumed.step_results
    assert tool.call_count == 1


def test_synthesis_failure_after_approval_does_not_execute_tool_twice(tmp_path):
    async def scenario():
        registry, tool = registry_with_spy()
        synthesis = FakeGroundedAnswerService(
            GroundedAnswerError("synthesis failed")
        )
        async with async_checkpoint_saver(tmp_path / "failure.sqlite") as saver:
            service = ApprovalWorkflowService(
                FakePlanner(approval_plan()), registry, saver, synthesis
            )
            await service.start("Create an issue", thread_id="thread-failure")
            with pytest.raises(GroundedAnswerError, match="synthesis failed"):
                await service.resume("thread-failure", "approve")
        return synthesis, tool

    synthesis, tool = asyncio.run(scenario())
    assert len(synthesis.calls) == 1
    assert tool.call_count == 1
