from typing import Literal

from app.graph.state import AgentState
from app.services.planner_service import PlanningError


PlanRoute = Literal["execute", "approval", "end"]


def route_plan_step(state: AgentState) -> PlanRoute:
    plan = state.get("plan")
    if plan is None:
        raise PlanningError("Execution plan is required")

    current_step_index = state.get("current_step_index", 0)
    if current_step_index < 0:
        raise PlanningError("Current step index must not be negative")
    if current_step_index >= len(plan.steps):
        return "end"

    current_step = plan.steps[current_step_index]
    if current_step.requires_approval:
        return "approval"
    return "execute"
