from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

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


class ApprovalRunNotFoundError(Exception):
    """Raised when an approval run audit record does not exist."""


class ApprovalRunThreadMismatchError(Exception):
    """Raised when a run is not bound to the supplied approval thread."""


class ApprovalRunNotPendingError(Exception):
    """Raised when an approval run is not awaiting a decision."""


@dataclass(slots=True)
class PersistentApprovalWorkflowResult:
    run_id: str
    thread_id: str
    plan: ExecutionPlan
    status: ApprovalWorkflowStatus
    current_step_index: int
    step_results: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None


WorkflowFailure = (
    PlanningError,
    LLMProviderError,
    ToolExecutionError,
    ApprovalThreadConflictError,
    ApprovalNotPendingError,
    ApprovalThreadNotFoundError,
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

    @staticmethod
    def _safe_result(result: ApprovalWorkflowResult) -> dict[str, Any]:
        return {
            "thread_id": result.thread_id,
            "current_step_index": result.current_step_index,
            "step_results": result.step_results,
            "pending_approval": result.pending_approval,
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
        )
