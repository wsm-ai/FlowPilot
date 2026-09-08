from typing import Any, Awaitable, Callable

from langchain_core.runnables import RunnableConfig

from app.graph.state import AgentState
from app.services.planner_service import PlanningError
from app.reliability.side_effects import (
    SideEffectExecutionError,
    SideEffectExecutor,
)
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


def create_approved_plan_step_executor(
    registry: ToolRegistry,
    side_effect_executor: SideEffectExecutor | None = None,
) -> PlanStepExecutor:
    async def execute_approved_step(
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            raise PlanningError("Execution plan is required")

        current_step_index = state.get("current_step_index", 0)
        if current_step_index < 0 or current_step_index >= len(plan.steps):
            raise PlanningError("Current step index is outside the execution plan")

        current_step = plan.steps[current_step_index]
        if not current_step.requires_approval:
            raise PlanningError("Current plan step does not require approval")
        if state.get("approval_decision") != "approve":
            raise PlanningError("Approved execution requires an approve decision")
        if state.get("pending_approval") is not None:
            raise PlanningError("Pending approval must be cleared before execution")

        if side_effect_executor is None:
            raise SideEffectExecutionError(
                "Side-effect execution protection is required"
            )
        configurable = {} if config is None else config.get("configurable", {})
        run_id = configurable.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise SideEffectExecutionError(
                "Side-effect execution identity is required"
            )
        result = await side_effect_executor.execute(
            run_id=run_id,
            step_id=current_step.id,
            action=current_step.action,
            arguments=current_step.arguments,
            operation=lambda: registry.execute(
                current_step.action,
                current_step.arguments,
            ),
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

    return execute_approved_step
