from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from app.grounding.lifecycle import GroundingSynthesisStatus
from app.grounding.models import GroundedAnswer
from app.persistence.models import create_agent_run_record
from app.persistence.repository import RunRepository
from app.providers.base import LLMProviderError
from app.schemas.planning import ExecutionPlan
from app.services.approval_workflow_service import (
    ApprovalNotPendingError,
    ApprovalThreadConflictError,
    ApprovalThreadNotFoundError,
    ApprovalWorkflowResult,
    ApprovalWorkflowService,
    ApprovalWorkflowStatus,
)
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError
from app.reliability.side_effects import SideEffectExecutionError


class ApprovalRunNotFoundError(Exception):
    """Raised when an approval run audit record does not exist."""


class ApprovalRunThreadMismatchError(Exception):
    """Raised when a run is not bound to the supplied approval thread."""


class ApprovalRunNotPendingError(Exception):
    """Raised when an approval run is not awaiting a decision."""


class GroundedAnswerRetryNotAllowedError(Exception):
    """Raised when a run is not eligible for grounded answer retry."""


class GroundedAnswerRetryStateError(Exception):
    """Raised when persisted data cannot safely support answer retry."""


@dataclass(slots=True)
class PersistentApprovalWorkflowResult:
    run_id: str
    thread_id: str
    plan: ExecutionPlan
    status: ApprovalWorkflowStatus
    current_step_index: int
    step_results: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    grounded_answer: GroundedAnswer | None = None
    grounding_status: GroundingSynthesisStatus = "not_attempted"
    grounding_error_type: str | None = None


WorkflowFailure = (
    PlanningError,
    LLMProviderError,
    ToolExecutionError,
    ApprovalThreadConflictError,
    ApprovalNotPendingError,
    ApprovalThreadNotFoundError,
    SideEffectExecutionError,
)


class PersistentApprovalWorkflowService:
    def __init__(
        self,
        workflow_service: ApprovalWorkflowService,
        run_repository: RunRepository,
    ) -> None:
        self._workflow_service = workflow_service
        self._run_repository = run_repository

    async def start(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
    ) -> PersistentApprovalWorkflowResult:
        resolved_thread_id = thread_id if thread_id is not None else str(uuid4())
        record = create_agent_run_record(
            mode="planned",
            input_text=goal,
            thread_id=resolved_thread_id,
        )
        await self._run_repository.create(record)

        try:
            workflow_result = await self._workflow_service.start(
                goal,
                thread_id=resolved_thread_id,
            )
        except WorkflowFailure as exc:
            await self._run_repository.update(
                record.run_id,
                status="failed",
                error_type=type(exc).__name__,
            )
            raise

        await self._run_repository.update(
            record.run_id,
            status=workflow_result.status,
            result=self._safe_result(workflow_result),
        )
        return self._to_result(record.run_id, workflow_result)

    async def resume(
        self,
        run_id: str,
        thread_id: str,
        decision: Literal["approve", "reject"],
    ) -> PersistentApprovalWorkflowResult:
        record = await self._run_repository.get(run_id)
        if record is None:
            raise ApprovalRunNotFoundError("Approval run not found")
        if record.mode != "planned" or record.thread_id != thread_id:
            raise ApprovalRunThreadMismatchError(
                "Run does not match approval thread"
            )
        if record.status != "approval_required":
            raise ApprovalRunNotPendingError("Approval run is not pending")

        await self._run_repository.update(run_id, status="running")
        try:
            workflow_result = await self._workflow_service.resume(
                thread_id,
                decision,
                run_id=run_id,
            )
        except WorkflowFailure as exc:
            await self._run_repository.update(
                run_id,
                status="failed",
                error_type=type(exc).__name__,
            )
            raise

        await self._run_repository.update(
            run_id,
            status=workflow_result.status,
            result=self._safe_result(workflow_result),
        )
        return self._to_result(run_id, workflow_result)

    async def retry_grounded_answer(
        self,
        run_id: str,
        thread_id: str,
    ) -> PersistentApprovalWorkflowResult:
        record = await self._run_repository.get(run_id)
        if record is None:
            raise ApprovalRunNotFoundError("Approval run not found")
        if record.mode != "planned" or record.thread_id != thread_id:
            raise ApprovalRunThreadMismatchError(
                "Run does not match approval thread"
            )
        if record.status != "completed":
            raise GroundedAnswerRetryNotAllowedError(
                "Grounded answer retry is not allowed"
            )

        stored = record.result
        if not isinstance(stored, dict):
            raise GroundedAnswerRetryStateError(
                "Grounded answer retry state is invalid"
            )
        if stored.get("grounding_status") != "failed":
            raise GroundedAnswerRetryNotAllowedError(
                "Grounded answer retry is not allowed"
            )
        try:
            plan = ExecutionPlan.model_validate(stored.get("plan"))
        except Exception as exc:
            raise GroundedAnswerRetryStateError(
                "Grounded answer retry state is invalid"
            ) from exc
        step_results = stored.get("step_results")
        current_step_index = stored.get("current_step_index")
        if (
            not isinstance(step_results, list)
            or not all(isinstance(item, dict) for item in step_results)
            or not isinstance(current_step_index, int)
        ):
            raise GroundedAnswerRetryStateError(
                "Grounded answer retry state is invalid"
            )

        outcome = await self._workflow_service.retry_grounded_answer(
            goal=record.input_text,
            plan=plan,
            step_results=step_results,
        )
        workflow_result = ApprovalWorkflowResult(
            thread_id=thread_id,
            plan=plan,
            status="completed",
            current_step_index=current_step_index,
            step_results=[item.copy() for item in step_results],
            pending_approval=None,
            grounded_answer=outcome.grounded_answer,
            grounding_status=outcome.status,
            grounding_error_type=outcome.error_type,
        )
        await self._run_repository.update(
            run_id,
            status="completed",
            result=self._safe_result(workflow_result),
        )
        return self._to_result(run_id, workflow_result)

    @staticmethod
    def _safe_result(result: ApprovalWorkflowResult) -> dict[str, Any]:
        return {
            "thread_id": result.thread_id,
            "plan": result.plan.model_dump(mode="json"),
            "current_step_index": result.current_step_index,
            "step_results": result.step_results,
            "pending_approval": result.pending_approval,
            "grounded_answer": (
                None
                if result.grounded_answer is None
                else result.grounded_answer.model_dump(mode="json")
            ),
            "grounding_status": result.grounding_status,
            "grounding_error_type": result.grounding_error_type,
        }

    @staticmethod
    def _to_result(
        run_id: str,
        result: ApprovalWorkflowResult,
    ) -> PersistentApprovalWorkflowResult:
        return PersistentApprovalWorkflowResult(
            run_id=run_id,
            thread_id=result.thread_id,
            plan=result.plan,
            status=result.status,
            current_step_index=result.current_step_index,
            step_results=result.step_results,
            pending_approval=result.pending_approval,
            grounded_answer=result.grounded_answer,
            grounding_status=result.grounding_status,
            grounding_error_type=result.grounding_error_type,
        )
