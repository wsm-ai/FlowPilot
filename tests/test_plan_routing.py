import asyncio

import pytest

from app.graph.routing import route_plan_step
from app.graph.routing_workflow import create_plan_routing_graph
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.planner_service import PlanningError


def create_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Review feedback and prepare an issue",
        steps=[
            PlanStep(
                id=1,
                description="Review customer feedback",
                action="get_customer_feedback",
                requires_approval=False,
            ),
            PlanStep(
                id=2,
                description="Prepare an issue",
                action="prepare_github_issue",
                requires_approval=True,
            ),
        ],
    )


def run_graph(plan: ExecutionPlan, current_step_index: int):
    graph = create_plan_routing_graph()
    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [],
                "llm_response": None,
                "answer": None,
                "executed_tools": [],
                "plan": plan,
                "current_step_index": current_step_index,
                "route": None,
            }
        )
    )
    return graph, result


def test_plan_routing_graph_compiles():
    graph = create_plan_routing_graph()

    assert graph is not None
    assert {"router_entry", "execute_marker", "approval_marker"}.issubset(
        graph.get_graph().nodes
    )


def test_non_approval_step_routes_to_execute():
    _, result = run_graph(create_plan(), 0)

    assert result["route"] == "execute"
    assert result["current_step_index"] == 0
    assert result["executed_tools"] == []


def test_approval_step_routes_to_approval():
    _, result = run_graph(create_plan(), 1)

    assert result["route"] == "approval"
    assert result["current_step_index"] == 1
    assert result["executed_tools"] == []


@pytest.mark.parametrize("current_step_index", [2, 3])
def test_exhausted_plan_routes_directly_to_end(current_step_index: int):
    _, result = run_graph(create_plan(), current_step_index)

    assert result["route"] == "end"
    assert result["current_step_index"] == current_step_index
    assert result["executed_tools"] == []


def test_router_reads_the_second_step():
    state = {
        "plan": create_plan(),
        "current_step_index": 1,
    }

    assert route_plan_step(state) == "approval"


def test_missing_plan_raises_planning_error():
    graph = create_plan_routing_graph()

    with pytest.raises(PlanningError, match="Execution plan is required"):
        asyncio.run(
            graph.ainvoke(
                {
                    "messages": [],
                    "llm_response": None,
                    "answer": None,
                    "executed_tools": [],
                    "current_step_index": 0,
                    "route": None,
                }
            )
        )
