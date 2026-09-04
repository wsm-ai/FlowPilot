import pytest
from fastapi.testclient import TestClient

from app.api.agent import get_planned_agent_service
from app.main import app
from app.providers.base import LLMProviderError
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.planned_agent_service import PlannedAgentResult
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


class FakePlannedAgentService:
    async def run(self, goal: str) -> PlannedAgentResult:
        return PlannedAgentResult(
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
        )


@pytest.fixture
def client():
    app.dependency_overrides[get_planned_agent_service] = FakePlannedAgentService
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_planned_agent_service, None)


def test_plan_run_returns_structured_result(client: TestClient):
    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Review feedback"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["plan"]["goal"] == "Review feedback"
    assert body["status"] == "completed"
    assert body["current_step_index"] == 1
    assert body["step_results"][0]["result"] == [
        {"id": "FB-001"},
        {"id": "FB-003"},
    ]


@pytest.mark.parametrize("goal", ["", "   "])
def test_plan_run_rejects_empty_or_blank_goal(client: TestClient, goal: str):
    response = client.post("/api/v1/agent/plan-run", json={"goal": goal})

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
        async def run(self, goal: str) -> PlannedAgentResult:
            raise error

    app.dependency_overrides[get_planned_agent_service] = FailingService

    response = client.post(
        "/api/v1/agent/plan-run",
        json={"goal": "Review feedback"},
    )

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert str(error) not in response.text
