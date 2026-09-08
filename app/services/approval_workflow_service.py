from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from app.graph.approval_workflow import create_approval_graph
from app.grounding.models import GroundedAnswer
from app.grounding.lifecycle import (
    GroundingSynthesisOutcome,
    GroundingSynthesisStatus,
    attempt_grounded_synthesis,
)
from app.schemas.planning import ExecutionPlan
from app.reliability.side_effects import SideEffectExecutor
from app.services.grounded_answer_service import GroundedAnswerService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


ApprovalWorkflowStatus = Literal[
    "completed",
    "approval_required",
    "rejected",
]


class ApprovalThreadNotFoundError(Exception):
    """Raised when an approval thread has no checkpoint state."""


class ApprovalThreadConflictError(Exception):
    """Raised when a new plan would reuse an existing approval thread."""


class ApprovalNotPendingError(Exception):
    """Raised when a thread is not currently waiting for approval."""


@dataclass(slots=True)
class ApprovalWorkflowResult:
    thread_id: str
    plan: ExecutionPlan
    status: ApprovalWorkflowStatus
    current_step_index: int
    step_results: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    grounded_answer: GroundedAnswer | None = None
    grounding_status: GroundingSynthesisStatus = "not_attempted"
    grounding_error_type: str | None = None


class ApprovalWorkflowService:
    def __init__(
        self,
        planner_service: PlannerService,
        registry: ToolRegistry,
        checkpointer: BaseCheckpointSaver,
        grounded_answer_service: GroundedAnswerService | None = None,
        side_effect_executor: SideEffectExecutor | None = None,
    ) -> None:
        self._planner_service = planner_service
        self._graph = create_approval_graph(
            registry, checkpointer, side_effect_executor
        )
        self._grounded_answer_service = grounded_answer_service

    async def start(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
    ) -> ApprovalWorkflowResult:
        resolved_thread_id = thread_id if thread_id is not None else str(uuid4())
        config = self._config(resolved_thread_id)

        existing_snapshot = await self._graph.aget_state(config)
        if existing_snapshot.values:
            raise ApprovalThreadConflictError("Approval thread already exists")

        plan = await self._planner_service.create_plan(goal)
        await self._graph.ainvoke(
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
                "pending_approval": None,
                "approval_decision": None,
            },
            config=config,
        )
        snapshot = await self._graph.aget_state(config)
        result = self._build_result(resolved_thread_id, snapshot)
        return await self._synthesize_if_completed(result, goal)

    async def resume(
        self,
        thread_id: str,
        decision: Literal["approve", "reject"],
        *,
        run_id: str | None = None,
    ) -> ApprovalWorkflowResult:
        config = self._config(thread_id, run_id=run_id)
        snapshot = await self._graph.aget_state(config)
        if not snapshot.values:
            raise ApprovalThreadNotFoundError("Approval thread not found")
        if not isinstance(snapshot.values.get("pending_approval"), dict):
            raise ApprovalNotPendingError("No pending approval for this thread")

        await self._graph.ainvoke(
            Command(resume={"decision": decision}),
            config=config,
        )
        snapshot = await self._graph.aget_state(config)
        result = self._build_result(thread_id, snapshot)
        goal = snapshot.values.get("goal")
        if not isinstance(goal, str):
            raise PlanningError("Approval workflow returned an invalid goal")
        return await self._synthesize_if_completed(result, goal)

    async def _synthesize_if_completed(
        self,
        result: ApprovalWorkflowResult,
        goal: str,
    ) -> ApprovalWorkflowResult:
        if result.status == "completed":
            outcome = await self.retry_grounded_answer(
                goal=goal,
                plan=result.plan,
                step_results=result.step_results,
            )
            result.grounded_answer = outcome.grounded_answer
            result.grounding_status = outcome.status
            result.grounding_error_type = outcome.error_type
        return result

    async def retry_grounded_answer(
        self,
        *,
        goal: str,
        plan: ExecutionPlan,
        step_results: list[dict[str, Any]],
    ) -> GroundingSynthesisOutcome:
        return await attempt_grounded_synthesis(
            self._grounded_answer_service,
            goal=goal,
            plan=plan,
            step_results=step_results,
        )

    @staticmethod
    def _config(
        thread_id: str, *, run_id: str | None = None
    ) -> dict[str, dict[str, str]]:
        configurable = {"thread_id": thread_id}
        if run_id is not None:
            configurable["run_id"] = run_id
        return {"configurable": configurable}

    @staticmethod
    def _build_result(thread_id: str, snapshot: Any) -> ApprovalWorkflowResult:
        values = snapshot.values
        try:
            plan = ExecutionPlan.model_validate(values.get("plan"))
        except Exception as exc:
            raise PlanningError("Approval workflow returned an invalid plan") from exc

        current_step_index = values.get("current_step_index")
        if not isinstance(current_step_index, int):
            raise PlanningError("Approval workflow returned an invalid step index")

        raw_step_results = values.get("step_results", [])
        if not isinstance(raw_step_results, list) or not all(
            isinstance(record, dict) for record in raw_step_results
        ):
            raise ToolExecutionError("Invalid plan step results")
        step_results = [record.copy() for record in raw_step_results]

        pending_approval = values.get("pending_approval")
        graph_ended = not snapshot.next
        if isinstance(pending_approval, dict):
            status: ApprovalWorkflowStatus = "approval_required"
            pending_result = pending_approval.copy()
        elif values.get("approval_decision") == "reject" and graph_ended:
            status = "rejected"
            pending_result = None
        elif current_step_index >= len(plan.steps) and graph_ended:
            status = "completed"
            pending_result = None
        else:
            raise PlanningError("Approval workflow ended in an unknown state")

        return ApprovalWorkflowResult(
            thread_id=thread_id,
            plan=plan,
            status=status,
            current_step_index=current_step_index,
            step_results=step_results,
            pending_approval=pending_result,
        )
