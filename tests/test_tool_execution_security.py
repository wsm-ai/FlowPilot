import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.graph.execution_nodes import create_approved_plan_step_executor
from app.reliability.side_effects import SideEffectExecutionError
from app.schemas.planning import ExecutionPlan, PlanStep
from app.security.rbac import Role
from app.security.api_key import APIKeyAuthenticationMiddleware
from app.security.rbac import APIKeyRoleBinding
from app.security.tool_authorization import (
    ApprovalGrantAuthority,
    ApprovedToolExecutionGrant,
    ToolRisk,
    tool_execution_role,
)
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


class SpyTool:
    name = "controlled_tool"
    description = "A controlled test tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments):
        self.calls += 1
        return {"ok": True}


class SpySideEffectExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, *, operation, **kwargs):
        self.calls += 1
        return await operation()


TEST_APPROVAL_AUTHORITY = ApprovalGrantAuthority()


def execute(registry: ToolRegistry):
    return asyncio.run(registry.execute("controlled_tool", {}))


def grant(action="controlled_tool", arguments=None, **overrides):
    values = {
        "run_id": "run-secure",
        "thread_id": "thread-secure",
        "step_id": 1,
        "action": action,
        "arguments": {} if arguments is None else arguments,
    }
    values.update(overrides)
    return TEST_APPROVAL_AUTHORITY.issue(**values)


@pytest.mark.parametrize("role", [Role.ADMIN, Role.OPERATOR])
def test_authorized_roles_can_execute_read_only_tool(role):
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.READ_ONLY)

    with tool_execution_role(role):
        assert execute(registry) == {"ok": True}
    assert tool.calls == 1


def test_viewer_cannot_execute_even_read_only_tool_directly():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.READ_ONLY)

    with tool_execution_role(Role.VIEWER), pytest.raises(
        ToolExecutionError, match="not authorized"
    ):
        execute(registry)
    assert tool.calls == 0


def test_unclassified_tool_is_denied_before_dispatch():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool)

    with tool_execution_role(Role.ADMIN), pytest.raises(
        ToolExecutionError, match="not authorized"
    ):
        execute(registry)
    assert tool.calls == 0

    with tool_execution_role(Role.ADMIN), TEST_APPROVAL_AUTHORITY.activate(grant()), pytest.raises(
        ToolExecutionError, match="not authorized"
    ):
        execute(registry)
    assert tool.calls == 0


def test_high_risk_tool_requires_exact_server_approval_and_admin():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)

    with tool_execution_role(Role.ADMIN), pytest.raises(
        ToolExecutionError, match="approval is required"
    ):
        execute(registry)
    with tool_execution_role(Role.ADMIN), TEST_APPROVAL_AUTHORITY.activate(grant("other_tool")), pytest.raises(
        ToolExecutionError, match="approval is required"
    ):
        execute(registry)
    with tool_execution_role(Role.OPERATOR), TEST_APPROVAL_AUTHORITY.activate(grant(tool.name)), pytest.raises(
        ToolExecutionError, match="not authorized"
    ):
        execute(registry)
    assert tool.calls == 0

    with tool_execution_role(Role.ADMIN), TEST_APPROVAL_AUTHORITY.activate(grant(tool.name)):
        assert execute(registry) == {"ok": True}
    assert tool.calls == 1


def test_registry_rejects_approved_action_with_different_arguments():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)

    with tool_execution_role(Role.ADMIN), TEST_APPROVAL_AUTHORITY.activate(
        grant(tool.name, arguments={"approved": True})
    ), pytest.raises(ToolExecutionError, match="does not match arguments"):
        asyncio.run(registry.execute(tool.name, {"approved": False}))

    assert tool.calls == 0


def test_registry_accepts_approved_action_with_matching_arguments():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)
    arguments = {"approved": True, "nested": {"value": 1}}

    with tool_execution_role(Role.ADMIN), TEST_APPROVAL_AUTHORITY.activate(
        grant(tool.name, arguments=arguments)
    ):
        result = asyncio.run(registry.execute(tool.name, arguments))

    assert result == {"ok": True}
    assert tool.calls == 1


def test_approved_executor_checks_authorization_before_ledger_dispatch():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)
    side_effects = SpySideEffectExecutor()
    executor = create_approved_plan_step_executor(registry, side_effects)
    state = {
        "plan": ExecutionPlan(
            goal="test",
            steps=[
                PlanStep(
                    id=1,
                    description="Execute controlled tool",
                    action=tool.name,
                    arguments={},
                    requires_approval=True,
                )
            ],
        ),
        "current_step_index": 0,
        "approval_decision": "approve",
        "pending_approval": None,
        "step_results": [],
    }

    with tool_execution_role(Role.OPERATOR), pytest.raises(
        SideEffectExecutionError, match="grant is invalid"
    ):
        asyncio.run(
            executor(state, {"configurable": {"run_id": "run-secure"}})
        )
    assert side_effects.calls == 0
    assert tool.calls == 0


@pytest.mark.parametrize(
    "grant_override",
    [
        {"run_id": "other-run"},
        {"thread_id": "other-thread"},
        {"step_id": 2},
        {"action": "other_tool"},
        {"arguments": {"changed": True}},
    ],
)
def test_approved_executor_rejects_grant_bound_to_other_operation(grant_override):
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)
    side_effects = SpySideEffectExecutor()
    executor = create_approved_plan_step_executor(registry, side_effects)
    state = {
        "plan": ExecutionPlan(
            goal="test",
            steps=[PlanStep(
                id=1,
                description="Execute controlled tool",
                action=tool.name,
                arguments={},
                requires_approval=True,
            )],
        ),
        "current_step_index": 0,
        "approval_decision": "approve",
        "pending_approval": None,
        "step_results": [],
    }
    execution_config = {
        "configurable": {
            "run_id": "run-secure",
            "thread_id": "thread-secure",
        }
    }

    with TEST_APPROVAL_AUTHORITY.activate(grant(**grant_override)), pytest.raises(
        SideEffectExecutionError, match="grant is invalid"
    ):
        asyncio.run(executor(state, execution_config))
    assert side_effects.calls == 0
    assert tool.calls == 0


def test_valid_admin_grant_executes_once_through_ledger_boundary():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)
    side_effects = SpySideEffectExecutor()
    executor = create_approved_plan_step_executor(registry, side_effects)
    state = {
        "plan": ExecutionPlan(
            goal="test",
            steps=[PlanStep(
                id=1,
                description="Execute controlled tool",
                action=tool.name,
                arguments={},
                requires_approval=True,
            )],
        ),
        "current_step_index": 0,
        "approval_decision": "approve",
        "pending_approval": None,
        "step_results": [],
    }
    execution_config = {
        "configurable": {
            "run_id": "run-secure",
            "thread_id": "thread-secure",
        }
    }

    with tool_execution_role(Role.ADMIN), TEST_APPROVAL_AUTHORITY.activate(grant()):
        result = asyncio.run(executor(state, execution_config))

    assert result["current_step_index"] == 1
    assert side_effects.calls == 1
    assert tool.calls == 1


def test_execution_role_context_is_restored():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)

    with tool_execution_role(Role.ADMIN), TEST_APPROVAL_AUTHORITY.activate(grant(tool.name)):
        execute(registry)

    with pytest.raises(ToolExecutionError, match="approval is required"):
        execute(registry)


def test_authenticated_role_reaches_tool_boundary_and_blocks_before_dispatch():
    read_registry = ToolRegistry()
    read_tool = SpyTool()
    read_registry.register(read_tool, risk=ToolRisk.READ_ONLY)
    side_registry = ToolRegistry()
    side_tool = SpyTool()
    side_registry.register(side_tool, risk=ToolRisk.HIGH_RISK)

    class Settings:
        flowpilot_auth_enabled = True
        flowpilot_api_key = SecretStr("admin-key")
        flowpilot_api_keys = (
            APIKeyRoleBinding(key=SecretStr("operator-key"), role=Role.OPERATOR),
            APIKeyRoleBinding(key=SecretStr("viewer-key"), role=Role.VIEWER),
        )

    app = FastAPI()
    app.add_middleware(
        APIKeyAuthenticationMiddleware,
        settings_provider=Settings,
    )

    @app.post("/api/v1/agent/run")
    async def run_read_tool():
        return await read_registry.execute(read_tool.name, {})

    @app.post("/api/v1/agent/approval/resume")
    async def run_approved_tool():
        with TEST_APPROVAL_AUTHORITY.activate(grant(side_tool.name)):
            return await side_registry.execute(side_tool.name, {})

    with TestClient(app) as client:
        operator = client.post(
            "/api/v1/agent/run",
            headers={"Authorization": "Bearer operator-key", "X-Role": "admin"},
            json={"role": "admin"},
        )
        viewer = client.post(
            "/api/v1/agent/run",
            headers={"Authorization": "Bearer viewer-key"},
        )
        operator_side_effect = client.post(
            "/api/v1/agent/approval/resume",
            headers={"Authorization": "Bearer operator-key"},
        )
        admin_side_effect = client.post(
            "/api/v1/agent/approval/resume",
            headers={"Authorization": "Bearer admin-key"},
        )

    assert operator.status_code == 200
    assert viewer.status_code == 403
    assert operator_side_effect.status_code == 403
    assert admin_side_effect.status_code == 200
    assert read_tool.calls == 1
    assert side_tool.calls == 1
