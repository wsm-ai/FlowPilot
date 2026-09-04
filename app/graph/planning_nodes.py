from typing import Awaitable, Callable

from app.graph.state import AgentState
from app.services.planner_service import PlannerService, PlanningError


PlannerNode = Callable[[AgentState], Awaitable[dict[str, object]]]


def create_planner_node(planner_service: PlannerService) -> PlannerNode:
    async def planner_node(state: AgentState) -> dict[str, object]:
        goal = state.get("goal")
        if goal is None or not goal.strip():
            raise PlanningError("Planning goal is required")

        plan = await planner_service.create_plan(goal)
        return {"plan": plan}

    return planner_node


def create_execute_marker_node() -> PlannerNode:
    async def execute_marker_node(state: AgentState) -> dict[str, object]:
        return {"route": "execute"}

    return execute_marker_node


def create_approval_marker_node() -> PlannerNode:
    async def approval_marker_node(state: AgentState) -> dict[str, object]:
        return {"route": "approval"}

    return approval_marker_node
