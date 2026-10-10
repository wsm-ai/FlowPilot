import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.security import APIKeyAuthenticationMiddleware, RequestBodyLimitMiddleware
from app.security.rbac import Role
from app.security.tool_authorization import (
    ApprovalGrantAuthority,
    ToolRisk,
    tool_execution_role,
)
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


ADMIN_KEY = "security-integration-admin"
OPERATOR_KEY = "security-integration-operator"
VIEWER_KEY = "security-integration-viewer"


class SpyTool:
    name = "security_integration_tool"
    description = "Security integration test tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        return arguments


def _settings(*, body_limit: int = 1_048_576) -> Settings:
    return Settings(
        deepseek_api_key="offline-test-key",
        flowpilot_auth_enabled=True,
        flowpilot_api_key=ADMIN_KEY,
        flowpilot_api_keys=[
            {"key": OPERATOR_KEY, "role": "operator"},
            {"key": VIEWER_KEY, "role": "viewer"},
        ],
        flowpilot_max_request_body_bytes=body_limit,
        _env_file=None,
    )


def _authorization(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def test_ambiguous_credentials_never_reach_business_service():
    calls = 0
    app = FastAPI()
    app.add_middleware(
        APIKeyAuthenticationMiddleware,
        settings_provider=_settings,
    )

    @app.post("/api/v1/chat")
    async def protected_operation():
        nonlocal calls
        calls += 1
        return {"ok": True}

    attempts = [
        [("Authorization", f"Bearer {ADMIN_KEY}"), ("Authorization", f"Bearer {ADMIN_KEY}")],
        [("Authorization", f"Bearer {ADMIN_KEY},Bearer {ADMIN_KEY}")],
        [("Authorization", f"Bearer\t{ADMIN_KEY}\textra")],
    ]
    with TestClient(app) as client:
        responses = [
            client.post("/api/v1/chat", headers=headers)
            for headers in attempts
        ]
        query_response = client.post(
            f"/api/v1/chat?api_key={ADMIN_KEY}&role=admin"
        )

    assert [response.status_code for response in responses] == [401, 401, 401]
    assert query_response.status_code == 401
    assert calls == 0


@pytest.mark.parametrize(
    "path",
    [
        "/internal/sensitive-operation",
        "/internal/sensitive-operation/",
        "/internal%2Fsensitive-operation",
    ],
)
def test_unmapped_non_api_paths_remain_admin_only_before_dispatch(path: str):
    calls = 0
    app = FastAPI()
    app.add_middleware(
        APIKeyAuthenticationMiddleware,
        settings_provider=_settings,
    )

    @app.api_route(
        "/internal/sensitive-operation",
        methods=["GET", "POST"],
    )
    async def sensitive_operation():
        nonlocal calls
        calls += 1
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get(path, headers=_authorization(OPERATOR_KEY))

    assert response.status_code == 403
    assert calls == 0


def test_request_body_limit_and_authentication_fail_before_business_dispatch():
    calls = 0
    settings = _settings(body_limit=32)
    app = FastAPI()
    app.add_middleware(
        RequestBodyLimitMiddleware,
        settings_provider=lambda: settings,
    )
    app.add_middleware(
        APIKeyAuthenticationMiddleware,
        settings_provider=lambda: settings,
    )

    @app.post("/api/v1/chat")
    async def protected_operation():
        nonlocal calls
        calls += 1
        return {"ok": True}

    oversized = json.dumps({"message": "x" * 64})
    with TestClient(app) as client:
        unauthenticated = client.post("/api/v1/chat", content=oversized)
        authenticated = client.post(
            "/api/v1/chat",
            headers=_authorization(ADMIN_KEY),
            content=oversized,
        )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 413
    assert calls == 0


def test_concurrent_http_roles_do_not_cross_authorization_contexts():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.READ_ONLY)
    app = FastAPI()
    app.add_middleware(
        APIKeyAuthenticationMiddleware,
        settings_provider=_settings,
    )

    @app.post("/api/v1/chat")
    async def execute_read_only_tool():
        return await registry.execute(tool.name, {})

    async def run_requests() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await asyncio.gather(
                client.post("/api/v1/chat", headers=_authorization(ADMIN_KEY)),
                client.post("/api/v1/chat", headers=_authorization(OPERATOR_KEY)),
                client.post("/api/v1/chat", headers=_authorization(VIEWER_KEY)),
            )

    responses = asyncio.run(run_requests())

    assert [response.status_code for response in responses] == [200, 200, 403]
    assert len(tool.calls) == 2


def test_concurrent_approval_grants_are_task_local_and_argument_bound():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)
    authority = ApprovalGrantAuthority()
    both_ready = asyncio.Event()
    ready_count = 0
    ready_lock = asyncio.Lock()

    async def execute_with_grant(run_id: str, value: str) -> dict[str, Any]:
        nonlocal ready_count
        arguments = {"value": value}
        grant = authority.issue(
            run_id=run_id,
            thread_id=f"thread-{value}",
            step_id=1,
            action=tool.name,
            arguments=arguments,
        )
        with tool_execution_role(Role.ADMIN), authority.activate(grant):
            async with ready_lock:
                ready_count += 1
                if ready_count == 2:
                    both_ready.set()
            await both_ready.wait()
            return await registry.execute(tool.name, arguments)

    async def run_concurrently() -> list[dict[str, Any]]:
        return await asyncio.gather(
            execute_with_grant("run-a", "a"),
            execute_with_grant("run-b", "b"),
        )

    results = asyncio.run(run_concurrently())

    assert results == [{"value": "a"}, {"value": "b"}]
    assert sorted(call["value"] for call in tool.calls) == ["a", "b"]


def test_approval_argument_substitution_is_rejected_before_dispatch():
    registry = ToolRegistry()
    tool = SpyTool()
    registry.register(tool, risk=ToolRisk.HIGH_RISK)
    authority = ApprovalGrantAuthority()
    grant = authority.issue(
        run_id="run-secure",
        thread_id="thread-secure",
        step_id=1,
        action=tool.name,
        arguments={"target": "approved"},
    )

    with tool_execution_role(Role.ADMIN), authority.activate(grant), pytest.raises(
        ToolExecutionError,
        match="does not match arguments",
    ):
        asyncio.run(registry.execute(tool.name, {"target": "substituted"}))

    assert tool.calls == []
