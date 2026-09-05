import pytest
from fastapi.testclient import TestClient

from app.api.agent import get_persistent_approval_workflow_service
from app.grounding.evidence import EvidenceExtractionError
from app.grounding.models import Citation, GroundedAnswer
from app.main import app
from app.persistence.repository import PersistenceError
from app.providers.base import LLMProviderError
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import (
    ApprovalNotPendingError,
    ApprovalThreadConflictError,
    ApprovalThreadNotFoundError,
)
from app.services.persistent_approval_workflow_service import (
    ApprovalRunNotFoundError,
    ApprovalRunNotPendingError,
    ApprovalRunThreadMismatchError,
    PersistentApprovalWorkflowResult,
)
from app.services.planner_service import PlanningError
from app.services.grounded_answer_service import GroundedAnswerError
from app.tools.base import ToolExecutionError


PLAN = ExecutionPlan(
    goal="Review feedback",
    steps=[
        PlanStep(
            id=1,
            description="Retrieve feedback",
            action="get_customer_feedback",
            arguments={"customer_id": "C001", "priority": "high"},
        )
    ],
)


class FakePersistentApprovalWorkflowService:
    async def start(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
    ) -> PersistentApprovalWorkflowResult:
        return PersistentApprovalWorkflowResult(
            run_id="run-123",
            thread_id=thread_id or "generated-thread",
            plan=PLAN,
            status="completed",
            current_step_index=1,
            step_results=[
                {
                    "step_id": 1,
                    "action": "get_customer_feedback",
                    "result": [{"id": "FB-001"}, {"id": "FB-003"}],
                }
            ],
            pending_approval=None,
            grounded_answer=GroundedAnswer(
                answer="Grounded feedback answer",
                citations=[
                    Citation(
                        citation_id="E1",
                        chunk_id="CH-1",
                        document_id="DOC-1",
                        source="support-handbook",
                        title="Support Guide",
                    )
                ],
            ),
        )

    async def resume(
        self, run_id: str, thread_id: str, decision: str
    ) -> PersistentApprovalWorkflowResult:
        return PersistentApprovalWorkflowResult(
            run_id=run_id,
            thread_id=thread_id,
            plan=PLAN,
            status="completed" if decision == "approve" else "rejected",
            current_step_index=1 if decision == "approve" else 0,
            step_results=[] if decision == "reject" else [{"step_id": 1}],
            pending_approval=None,
            grounded_answer=(
                GroundedAnswer(
                    answer="Approved grounded answer",
                    citations=[
                        Citation(
                            citation_id="E1",
                            chunk_id="CH-1",
                            document_id="DOC-1",
                            source="support-handbook",
                        )
                    ],
                )
                if decision == "approve"
                else None
            ),
        )


@pytest.fixture
def client():
    app.dependency_overrides[
        get_persistent_approval_workflow_service
    ] = FakePersistentApprovalWorkflowService
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_persistent_approval_workflow_service, None)


def test_plan_run_returns_structured_result(client: TestClient):
    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Review feedback"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-123"
    assert body["thread_id"] == "generated-thread"
    assert body["plan"]["goal"] == "Review feedback"
    assert body["status"] == "completed"
    assert body["current_step_index"] == 1
    assert body["step_results"][0]["result"] == [
        {"id": "FB-001"},
        {"id": "FB-003"},
    ]
    assert body["pending_approval"] is None
    assert body["answer"] == "Grounded feedback answer"
    assert body["citations"] == [
        {
            "citation_id": "E1",
            "chunk_id": "CH-1",
            "document_id": "DOC-1",
            "source": "support-handbook",
            "title": "Support Guide",
        }
    ]


def test_plan_run_pending_approval_has_no_grounded_answer(client: TestClient):
    pending_plan = ExecutionPlan(
        goal="Create issue",
        steps=[
            PlanStep(
                id=1,
                description="Create issue",
                action="create_test_issue",
                arguments={"title": "Login issue"},
                requires_approval=True,
            )
        ],
    )

    class PendingService:
        async def start(self, goal: str, *, thread_id=None):
            return PersistentApprovalWorkflowResult(
                run_id="run-pending",
                thread_id="approval-thread",
                plan=pending_plan,
                status="approval_required",
                current_step_index=0,
                step_results=[],
                pending_approval={
                    "type": "plan_step_approval",
                    "step_id": 1,
                    "description": "Create issue",
                    "action": "create_test_issue",
                    "arguments": {"title": "Login issue"},
                },
                grounded_answer=None,
            )

    app.dependency_overrides[get_persistent_approval_workflow_service] = PendingService

    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Create issue", "thread_id": "approval-thread"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-pending"
    assert body["thread_id"] == "approval-thread"
    assert body["status"] == "approval_required"
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["pending_approval"] is not None


@pytest.mark.parametrize("goal", ["", "   "])
def test_plan_run_rejects_empty_or_blank_goal(client: TestClient, goal: str):
    response = client.post("/api/v1/agent/plan-run", json={"goal": goal})

    assert response.status_code == 422


@pytest.mark.parametrize("thread_id", ["", "   ", "x" * 201])
def test_plan_run_rejects_invalid_thread_id(client: TestClient, thread_id: str):
    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Review feedback", "thread_id": thread_id},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (
            PlanningError("sensitive planning error"),
            400,
            "Unable to execute the requested plan",
        ),
        (
            LLMProviderError("sensitive provider error"),
            502,
            "The language model service is unavailable",
        ),
        (
            ToolExecutionError("sensitive tool error"),
            500,
            "Plan step execution failed",
        ),
        (
            PersistenceError("sensitive persistence error"),
            500,
            "Agent persistence failed",
        ),
        (
            GroundedAnswerError("sensitive synthesis E999"),
            502,
            "Unable to generate grounded answer",
        ),
        (
            EvidenceExtractionError("sensitive invalid evidence"),
            500,
            "Grounded answer evidence is invalid",
        ),
    ],
)
def test_plan_run_maps_errors_without_leaking_details(
    client: TestClient,
    error: Exception,
    status_code: int,
    detail: str,
):
    class FailingService:
        async def start(self, goal: str, *, thread_id=None):
            raise error

    app.dependency_overrides[get_persistent_approval_workflow_service] = FailingService

    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Review feedback"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert str(error) not in response.text


def test_plan_run_maps_thread_conflict_without_leaking_details(client: TestClient):
    class FailingService:
        async def start(self, goal: str, *, thread_id=None):
            raise ApprovalThreadConflictError("sensitive conflict")

    app.dependency_overrides[get_persistent_approval_workflow_service] = FailingService
    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Review feedback", "thread_id": "existing"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Approval thread already exists"}
    assert "sensitive conflict" not in response.text


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_approval_resume_returns_unified_response(client: TestClient, decision: str):
    response = client.post(
        "/api/v1/agent/approval/resume",
        json={
            "run_id": "run-123",
            "thread_id": "approval-thread",
            "decision": decision,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run-123"
    assert body["thread_id"] == "approval-thread"
    assert body["status"] == ("completed" if decision == "approve" else "rejected")
    assert body["pending_approval"] is None
    if decision == "approve":
        assert body["answer"] == "Approved grounded answer"
        assert body["citations"][0]["citation_id"] == "E1"
    else:
        assert body["answer"] is None
        assert body["citations"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"run_id": "run-123", "thread_id": "approval-thread", "decision": "maybe"},
        {"run_id": "run-123", "thread_id": "", "decision": "approve"},
        {"run_id": "run-123", "thread_id": "   ", "decision": "approve"},
        {"run_id": "", "thread_id": "approval-thread", "decision": "approve"},
        {"run_id": "   ", "thread_id": "approval-thread", "decision": "approve"},
        {"run_id": "x" * 201, "thread_id": "approval-thread", "decision": "approve"},
    ],
)
def test_approval_resume_rejects_invalid_request(client: TestClient, payload):
    response = client.post("/api/v1/agent/approval/resume", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (ApprovalThreadNotFoundError("sensitive missing"), 404, "Approval thread not found"),
        (ApprovalThreadConflictError("sensitive conflict"), 409, "Approval thread already exists"),
        (ApprovalNotPendingError("sensitive pending"), 409, "No pending approval for this thread"),
        (ApprovalRunNotFoundError("sensitive run"), 404, "Approval run not found"),
        (ApprovalRunThreadMismatchError("sensitive mismatch"), 409, "Run does not match approval thread"),
        (ApprovalRunNotPendingError("sensitive run status"), 409, "Approval run is not pending"),
        (PlanningError("sensitive planning"), 400, "Unable to execute the requested plan"),
        (ToolExecutionError("sensitive tool"), 500, "Plan step execution failed"),
        (LLMProviderError("sensitive llm"), 502, "The language model service is unavailable"),
        (PersistenceError("sensitive persistence"), 500, "Agent persistence failed"),
        (GroundedAnswerError("sensitive E999"), 502, "Unable to generate grounded answer"),
        (EvidenceExtractionError("sensitive evidence"), 500, "Grounded answer evidence is invalid"),
    ],
)
def test_approval_resume_maps_errors_without_leaking_details(
    client: TestClient,
    error: Exception,
    status_code: int,
    detail: str,
):
    class FailingService:
        async def resume(self, run_id: str, thread_id: str, decision: str):
            raise error

    app.dependency_overrides[get_persistent_approval_workflow_service] = FailingService
    response = client.post(
        "/api/v1/agent/approval/resume",
        json={
            "run_id": "run-123",
            "thread_id": "approval-thread",
            "decision": "approve",
        },
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert str(error) not in response.text
