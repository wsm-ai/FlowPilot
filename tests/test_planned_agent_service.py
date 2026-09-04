import asyncio

import pytest

from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.llm_service import LLMService
from app.services.planned_agent_service import PlannedAgentService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.base import ToolExecutionError
from app.tools.registry import create_default_tool_registry


def feedback_step(
    step_id: int,
    customer_id: str,
    *,
    requires_approval: bool = False,
    action: str = "get_customer_feedback",
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=f"Retrieve feedback for {customer_id}",
        action=action,
        arguments={"customer_id": customer_id, "priority": "high"},
        requires_approval=requires_approval,
    )


class FakePlannerService:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan

    async def create_plan(self, goal: str) -> ExecutionPlan:
        return self.plan


def service_for_plan(plan: ExecutionPlan) -> PlannedAgentService:
    return PlannedAgentService(
        planner_service=FakePlannerService(plan),
        registry=create_default_tool_registry(),
    )


def test_single_step_plan_completes_with_result():
    plan = ExecutionPlan(goal="Review feedback", steps=[feedback_step(1, "C001")])

    result = asyncio.run(service_for_plan(plan).run(plan.goal))

    assert result.status == "completed"
    assert result.current_step_index == 1
    assert len(result.step_results) == 1
    records = result.step_results[0]["result"]
    assert {record["id"] for record in records} == {"FB-001", "FB-003"}


def test_two_step_plan_completes_both_steps():
    plan = ExecutionPlan(
        goal="Review two customers",
        steps=[feedback_step(1, "C001"), feedback_step(2, "C002")],
    )

    result = asyncio.run(service_for_plan(plan).run(plan.goal))

    assert result.status == "completed"
    assert result.current_step_index == 2
    assert len(result.step_results) == 2


def test_approval_step_stops_without_execution():
    plan = ExecutionPlan(
        goal="Prepare an approved change",
        steps=[feedback_step(1, "C001", requires_approval=True)],
    )

    result = asyncio.run(service_for_plan(plan).run(plan.goal))

    assert result.status == "approval_required"
    assert result.current_step_index == 0
    assert result.step_results == []


def test_unknown_action_propagates_tool_error():
    plan = ExecutionPlan(
        goal="Run unknown action",
        steps=[feedback_step(1, "C001", action="unknown_action")],
    )

    with pytest.raises(ToolExecutionError):
        asyncio.run(service_for_plan(plan).run(plan.goal))


class FakeProvider:
    model = "test-model"

    def __init__(self, content: str | None) -> None:
        self.content = content

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        return LLMResponse(content=self.content)


@pytest.mark.parametrize(
    ("content", "error_type"),
    [("{invalid", PlanningError), (None, LLMProviderError)],
)
def test_planner_errors_propagate(content, error_type):
    service = PlannedAgentService(
        planner_service=PlannerService(LLMService(FakeProvider(content))),
        registry=create_default_tool_registry(),
    )

    with pytest.raises(error_type):
        asyncio.run(service.run("Review feedback"))
