from dataclasses import dataclass, field
from typing import Any, Literal

from app.graph.execution_workflow import create_plan_execution_graph
from app.grounding.models import GroundedAnswer
from app.schemas.planning import ExecutionPlan
from app.services.grounded_answer_service import GroundedAnswerService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


PlannedAgentStatus = Literal["completed", "approval_required"]


@dataclass(slots=True)
class PlannedAgentResult:
    plan: ExecutionPlan
    current_step_index: int
    status: PlannedAgentStatus
    step_results: list[dict[str, Any]] = field(default_factory=list)
    grounded_answer: GroundedAnswer | None = None


class PlannedAgentService:
    def __init__(
        self,
        planner_service: PlannerService,
        registry: ToolRegistry,
        grounded_answer_service: GroundedAnswerService | None = None,
    ) -> None:
        self._planner_service = planner_service
        self._execution_graph = create_plan_execution_graph(registry)
        self._grounded_answer_service = grounded_answer_service

    async def run(self, goal: str) -> PlannedAgentResult:
        plan = await self._planner_service.create_plan(goal)
        result = await self._execution_graph.ainvoke(
            {
                "messages": [],
                "llm_response": None,
                "answer": None,
                "executed_tools": [],
                "goal": goal,
                "plan": plan,
                "current_step_index": 0,
                "route": None,
                "step_results": [],
            }
        )

        current_step_index = result.get("current_step_index")
        if not isinstance(current_step_index, int):
            raise PlanningError("Plan execution returned an invalid step index")

        route = result.get("route")
        if route == "approval":
            status: PlannedAgentStatus = "approval_required"
        elif current_step_index >= len(plan.steps):
            status = "completed"
        else:
            raise PlanningError("Plan execution ended in an unknown state")

        planned_result = PlannedAgentResult(
            plan=plan,
            current_step_index=current_step_index,
            status=status,
            step_results=self._convert_step_results(result.get("step_results", [])),
        )
        if status == "completed" and self._grounded_answer_service is not None:
            planned_result.grounded_answer = (
                await self._grounded_answer_service.synthesize(
                    goal=goal,
                    plan=planned_result.plan,
                    step_results=planned_result.step_results,
                )
            )
        return planned_result

    @staticmethod
    def _convert_step_results(records: Any) -> list[dict[str, Any]]:
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ToolExecutionError("Invalid plan step results")
        return [record.copy() for record in records]
