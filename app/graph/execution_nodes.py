from typing import Any, Awaitable, Callable

from app.graph.state import AgentState
from app.services.planner_service import PlanningError
from app.tools.registry import ToolRegistry


PlanStepExecutor = Callable[[AgentState], Awaitable[dict[str, Any]]]


def create_plan_step_executor(registry: ToolRegistry) -> PlanStepExecutor:
    async def execute_step(state: AgentState) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            raise PlanningError("Execution plan is required")

        current_step_index = state.get("current_step_index", 0)
        if current_step_index < 0 or current_step_index >= len(plan.steps):
            raise PlanningError("Current step index is outside the execution plan")

        current_step = plan.steps[current_step_index]
        if current_step.requires_approval:
            raise PlanningError(
                "Approval-required step cannot be executed automatically"
            )

        result = await registry.execute(
            current_step.action,
            current_step.arguments,
        )
        step_results = list(state.get("step_results", []))
        step_results.append(
            {
                "step_id": current_step.id,
                "action": current_step.action,
                "result": result,
            }
        )
        return {
            "step_results": step_results,
            "current_step_index": current_step_index + 1,
            "route": None,
        }

    return execute_step
