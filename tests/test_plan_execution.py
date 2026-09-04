import asyncio

import pytest

from app.graph.execution_nodes import create_plan_step_executor
from app.graph.execution_workflow import create_plan_execution_graph
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError
from app.tools.registry import create_default_tool_registry


def plan_step(
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


def create_plan(*steps: PlanStep) -> ExecutionPlan:
    return ExecutionPlan(goal="Review customer feedback", steps=list(steps))


def initial_state(plan: ExecutionPlan, index: int = 0):
    return {
        "messages": [],
        "llm_response": None,
        "answer": None,
        "executed_tools": [],
        "plan": plan,
        "current_step_index": index,
        "route": None,
        "step_results": [],
    }


def test_single_step_executes_and_routes_to_end():
    graph = create_plan_execution_graph(create_default_tool_registry())
    plan = create_plan(plan_step(1, "C001"))

    result = asyncio.run(graph.ainvoke(initial_state(plan)))

    assert result["route"] == "end"
    assert result["current_step_index"] == 1
    assert len(result["step_results"]) == 1
    tool_result = result["step_results"][0]["result"]
    assert {record["id"] for record in tool_result} == {"FB-001", "FB-003"}
    assert result["executed_tools"] == []


def test_two_steps_execute_in_order_and_end():
    graph = create_plan_execution_graph(create_default_tool_registry())
    plan = create_plan(plan_step(1, "C001"), plan_step(2, "C002"))

    result = asyncio.run(graph.ainvoke(initial_state(plan)))

    assert result["current_step_index"] == 2
    assert result["route"] == "end"
    assert [item["step_id"] for item in result["step_results"]] == [1, 2]
    assert len(result["step_results"]) == 2


def test_approval_step_stops_without_execution():
    graph = create_plan_execution_graph(create_default_tool_registry())
    plan = create_plan(
        plan_step(
            1,
            "C001",
            requires_approval=True,
            action="unregistered_approval_action",
        )
    )

    result = asyncio.run(graph.ainvoke(initial_state(plan)))

    assert result["route"] == "approval"
    assert result["current_step_index"] == 0
    assert result["step_results"] == []


def test_executor_rejects_approval_required_step():
    executor = create_plan_step_executor(create_default_tool_registry())
    plan = create_plan(plan_step(1, "C001", requires_approval=True))

    with pytest.raises(
        PlanningError,
        match="Approval-required step cannot be executed automatically",
    ):
        asyncio.run(executor(initial_state(plan)))


def test_executor_propagates_unknown_action_error():
    executor = create_plan_step_executor(create_default_tool_registry())
    plan = create_plan(plan_step(1, "C001", action="unknown_action"))

    with pytest.raises(ToolExecutionError):
        asyncio.run(executor(initial_state(plan)))


def test_executor_rejects_missing_plan():
    executor = create_plan_step_executor(create_default_tool_registry())

    with pytest.raises(PlanningError, match="Execution plan is required"):
        asyncio.run(executor({}))


@pytest.mark.parametrize("index", [-1, 1, 2])
def test_executor_rejects_index_outside_plan(index: int):
    executor = create_plan_step_executor(create_default_tool_registry())
    plan = create_plan(plan_step(1, "C001"))

    with pytest.raises(PlanningError, match="outside the execution plan"):
        asyncio.run(executor(initial_state(plan, index)))


def test_plan_step_arguments_default_to_empty_dict():
    step = PlanStep(
        id=1,
        description="Generate a report",
        action="generate_report",
    )

    assert step.arguments == {}
