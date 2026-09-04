from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.agent import get_approval_workflow_service, router
from app.persistence.checkpoint import async_checkpoint_saver
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.tools.registry import ToolRegistry


class FakePlannerService:
    async def create_plan(self, goal: str) -> ExecutionPlan:
        return ExecutionPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id=1,
                    description="Create a test issue",
                    action="create_test_issue",
                    arguments={"title": "Critical login bug"},
                    requires_approval=True,
                )
            ],
        )


class SpyIssueTool:
    name = "create_test_issue"
    description = "Create a test issue"
    parameters = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    def __init__(self) -> None:
        self.call_count = 0

    async def execute(self, arguments):
        self.call_count += 1
        return {"issue_id": "TEST-001", **arguments}


def test_plan_run_and_approval_resume_share_durable_thread(tmp_path):
    database_path = tmp_path / "approval-api.sqlite"
    registry = ToolRegistry()
    tool = SpyIssueTool()
    registry.register(tool)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with async_checkpoint_saver(database_path) as saver:
            app.state.checkpointer = saver
            yield

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(router)

    def override_service(request: Request) -> ApprovalWorkflowService:
        return ApprovalWorkflowService(
            FakePlannerService(),
            registry,
            request.app.state.checkpointer,
        )

    test_app.dependency_overrides[get_approval_workflow_service] = override_service

    with TestClient(test_app) as client:
        started = client.post(
            "/api/v1/agent/plan-run",
            json={"goal": "Create bug issue"},
        )
        assert started.status_code == 200
        started_body = started.json()
        assert started_body["status"] == "approval_required"
        assert tool.call_count == 0

        resumed = client.post(
            "/api/v1/agent/approval/resume",
            json={
                "thread_id": started_body["thread_id"],
                "decision": "approve",
            },
        )

    assert resumed.status_code == 200
    resumed_body = resumed.json()
    assert resumed_body["status"] == "completed"
    assert resumed_body["thread_id"] == started_body["thread_id"]
    assert tool.call_count == 1
