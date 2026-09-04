import pytest
from fastapi.testclient import TestClient

from app.api.agent import get_approval_workflow_service
from app.main import app
from app.providers.base import LLMProviderError
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import (
    ApprovalNotPendingError,
    ApprovalThreadConflictError,
    ApprovalThreadNotFoundError,
    ApprovalWorkflowResult,
)
from app.services.planner_service import PlanningError
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


class FakeApprovalWorkflowService:
    async def start(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
    ) -> ApprovalWorkflowResult:
        return ApprovalWorkflowResult(
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
        )

    async def resume(self, thread_id: str, decision: str) -> ApprovalWorkflowResult:
        return ApprovalWorkflowResult(
            thread_id=thread_id,
            plan=PLAN,
            status="completed" if decision == "approve" else "rejected",
            current_step_index=1 if decision == "approve" else 0,
            step_results=[] if decision == "reject" else [{"step_id": 1}],
            pending_approval=None,
        )


@pytest.fixture
def client():
    app.dependency_overrides[
        get_approval_workflow_service
    ] = FakeApprovalWorkflowService
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_approval_workflow_service, None)


def test_plan_run_returns_structured_result(client: TestClient):
    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Review feedback"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "generated-thread"
    assert body["plan"]["goal"] == "Review feedback"
    assert body["status"] == "completed"
    assert body["current_step_index"] == 1
    assert body["step_results"][0]["result"] == [
        {"id": "FB-001"},
        {"id": "FB-003"},
    ]
    assert body["pending_approval"] is None


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

    app.dependency_overrides[get_approval_workflow_service] = FailingService

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

    app.dependency_overrides[get_approval_workflow_service] = FailingService
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
        json={"thread_id": "approval-thread", "decision": decision},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == "approval-thread"
    assert body["status"] == ("completed" if decision == "approve" else "rejected")
    assert body["pending_approval"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"thread_id": "approval-thread", "decision": "maybe"},
        {"thread_id": "", "decision": "approve"},
        {"thread_id": "   ", "decision": "approve"},
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
        (PlanningError("sensitive planning"), 400, "Unable to execute the requested plan"),
        (ToolExecutionError("sensitive tool"), 500, "Plan step execution failed"),
        (LLMProviderError("sensitive llm"), 502, "The language model service is unavailable"),
    ],
)
def test_approval_resume_maps_errors_without_leaking_details(
    client: TestClient,
    error: Exception,
    status_code: int,
    detail: str,
):
    class FailingService:
        async def resume(self, thread_id: str, decision: str):
            raise error

    app.dependency_overrides[get_approval_workflow_service] = FailingService
    response = client.post(
        "/api/v1/agent/approval/resume",
        json={"thread_id": "approval-thread", "decision": "approve"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert str(error) not in response.text
