from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import asyncio

from app.api.agent import get_persistent_approval_workflow_service, router
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.persistent_approval_workflow_service import (
    PersistentApprovalWorkflowService,
)
from app.tools.registry import ToolRegistry
from tests.support.side_effects import PASSTHROUGH_SIDE_EFFECT_EXECUTOR


_ApprovalWorkflowService = ApprovalWorkflowService


def ApprovalWorkflowService(*args, **kwargs):
    kwargs.setdefault("side_effect_executor", PASSTHROUGH_SIDE_EFFECT_EXECUTOR)
    return _ApprovalWorkflowService(*args, **kwargs)


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
    checkpoint_path = tmp_path / "approval-api-checkpoints.sqlite"
    repository = SQLiteRunRepository(tmp_path / "approval-api-runs.sqlite")
    registry = ToolRegistry()
    tool = SpyIssueTool()
    registry.register(tool)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await repository.initialize()
        async with async_checkpoint_saver(checkpoint_path) as saver:
            app.state.checkpointer = saver
            yield

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(router)

    def override_service(request: Request) -> PersistentApprovalWorkflowService:
        workflow = ApprovalWorkflowService(
            FakePlannerService(), registry, request.app.state.checkpointer
        )
        return PersistentApprovalWorkflowService(workflow, repository)

    test_app.dependency_overrides[
        get_persistent_approval_workflow_service
    ] = override_service

    with TestClient(test_app) as client:
        started = client.post(
            "/api/v1/agent/plan-run",
            json={"goal": "Create bug issue"},
        )
        assert started.status_code == 200
        started_body = started.json()
        assert started_body["status"] == "approval_required"
        assert tool.call_count == 0
        started_record = asyncio.run(repository.get(started_body["run_id"]))
        assert started_record.mode == "planned"
        assert started_record.thread_id == started_body["thread_id"]
        assert started_record.status == "approval_required"

        resumed = client.post(
            "/api/v1/agent/approval/resume",
            json={
                "run_id": started_body["run_id"],
                "thread_id": started_body["thread_id"],
                "decision": "approve",
            },
        )

    assert resumed.status_code == 200
    resumed_body = resumed.json()
    assert resumed_body["status"] == "completed"
    assert resumed_body["run_id"] == started_body["run_id"]
    assert resumed_body["thread_id"] == started_body["thread_id"]
    completed_record = asyncio.run(repository.get(started_body["run_id"]))
    assert completed_record.status == "completed"
    assert tool.call_count == 1
