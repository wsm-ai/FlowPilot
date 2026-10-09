import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.agent import (
    get_persistent_agent_service,
    get_persistent_approval_workflow_service,
)
from app.api.dependencies import get_llm_service
from app.core.config import Settings, get_settings
from app.main import app
from app.security import APIKeyAuthenticationMiddleware


ADMIN_KEY = "rbac-admin-key"
OPERATOR_KEY = "rbac-operator-key"
VIEWER_KEY = "rbac-viewer-key"


def authorization(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def rbac_app(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("FLOWPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("FLOWPILOT_API_KEY", ADMIN_KEY)
    monkeypatch.setenv(
        "FLOWPILOT_API_KEYS",
        json.dumps(
            [
                {"key": OPERATOR_KEY, "role": "operator"},
                {"key": VIEWER_KEY, "role": "viewer"},
            ]
        ),
    )
    get_settings.cache_clear()
    try:
        yield
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_admin_can_access_admin_documentation(rbac_app):
    with TestClient(app) as client:
        response = client.get("/openapi.json", headers=authorization(ADMIN_KEY))

    assert response.status_code == 200


def test_operator_can_use_chat_without_invoking_real_llm(rbac_app):
    class FakeLLMService:
        model = "fake-model"

        async def chat(self, message: str) -> str:
            return "fake reply"

    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers=authorization(OPERATOR_KEY),
            json={"message": "hello"},
        )

    assert response.status_code == 200
    assert response.json() == {"reply": "fake reply", "model": "fake-model"}


def test_operator_cannot_access_admin_documentation(rbac_app):
    with TestClient(app) as client:
        response = client.get("/openapi.json", headers=authorization(OPERATOR_KEY))

    assert response.status_code == 403
    assert response.json() == {"detail": "Insufficient permissions"}


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/v1/chat", {"message": "hello"}),
        ("/api/v1/agent/run", {"message": "hello"}),
        ("/api/v1/agent/plan-run", {"goal": "create issue"}),
        (
            "/api/v1/agent/approval/resume",
            {"run_id": "run-1", "thread_id": "thread-1", "decision": "approve"},
        ),
        (
            "/api/v1/agent/answer/retry",
            {"run_id": "run-1", "thread_id": "thread-1"},
        ),
    ],
)
def test_viewer_cannot_execute_business_operations(rbac_app, path, payload):
    with TestClient(app) as client:
        response = client.post(
            path,
            headers=authorization(VIEWER_KEY),
            json=payload,
        )

    assert response.status_code == 403


def test_operator_cannot_execute_approval_and_service_is_not_called(rbac_app):
    calls = 0

    class ForbiddenService:
        async def resume(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("approval service must not execute")

    app.dependency_overrides[
        get_persistent_approval_workflow_service
    ] = lambda: ForbiddenService()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/approval/resume",
            headers=authorization(OPERATOR_KEY),
            json={
                "run_id": "run-1",
                "thread_id": "thread-1",
                "decision": "approve",
            },
        )

    assert response.status_code == 403
    assert calls == 0


def test_viewer_cannot_trigger_agent_service(rbac_app):
    calls = 0

    class ForbiddenService:
        async def run(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("agent service must not execute")

    app.dependency_overrides[get_persistent_agent_service] = lambda: ForbiddenService()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/run",
            headers=authorization(VIEWER_KEY),
            json={"message": "hello"},
        )

    assert response.status_code == 403
    assert calls == 0


@pytest.mark.parametrize("key", [OPERATOR_KEY, VIEWER_KEY])
def test_non_admin_roles_cannot_access_mounted_mcp(rbac_app, key):
    with TestClient(app) as client:
        response = client.post("/mcp", headers=authorization(key), json={})

    assert response.status_code == 403


def test_missing_and_wrong_keys_still_return_401(rbac_app):
    with TestClient(app) as client:
        missing = client.post("/api/v1/chat", json={"message": "hello"})
        wrong = client.post(
            "/api/v1/chat",
            headers=authorization("wrong-key"),
            json={"message": "hello"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401


def test_client_supplied_role_header_cannot_elevate_viewer(rbac_app):
    headers = authorization(VIEWER_KEY) | {"X-Role": "admin"}
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/run",
            headers=headers,
            json={"message": "hello"},
        )

    assert response.status_code == 403


def test_client_supplied_json_role_cannot_elevate_viewer(rbac_app):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/run",
            headers=authorization(VIEWER_KEY),
            json={"message": "hello", "role": "admin"},
        )

    assert response.status_code == 403


def test_unknown_api_route_is_not_available_to_operator(rbac_app):
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/future-sensitive-route",
            headers=authorization(OPERATOR_KEY),
        )

    assert response.status_code == 403


@pytest.mark.parametrize("key", [OPERATOR_KEY, VIEWER_KEY])
def test_unmapped_non_api_route_is_admin_only_and_never_executes(key):
    calls = 0
    test_app = FastAPI()
    settings = Settings(
        deepseek_api_key="test-deepseek-key",
        flowpilot_auth_enabled=True,
        flowpilot_api_key=ADMIN_KEY,
        flowpilot_api_keys=[
            {"key": OPERATOR_KEY, "role": "operator"},
            {"key": VIEWER_KEY, "role": "viewer"},
        ],
        _env_file=None,
    )
    test_app.add_middleware(
        APIKeyAuthenticationMiddleware,
        settings_provider=lambda: settings,
    )

    @test_app.get("/internal/sensitive-operation")
    async def sensitive_operation():
        nonlocal calls
        calls += 1
        return {"status": "executed"}

    with TestClient(test_app) as client:
        response = client.get(
            "/internal/sensitive-operation",
            headers=authorization(key),
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "Insufficient permissions"}
    assert calls == 0


def test_health_remains_public_with_rbac_enabled(rbac_app):
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


@pytest.mark.parametrize(
    "bindings",
    [
        [{"key": "operator-key", "role": "owner"}],
        [{"key": "   ", "role": "operator"}],
        [
            {"key": "duplicate", "role": "operator"},
            {"key": "duplicate", "role": "viewer"},
        ],
    ],
)
def test_invalid_role_key_configuration_is_rejected(bindings):
    with pytest.raises(ValidationError):
        Settings(
            deepseek_api_key="test-deepseek-key",
            flowpilot_auth_enabled=True,
            flowpilot_api_key=None,
            flowpilot_api_keys=bindings,
            _env_file=None,
        )


def test_legacy_admin_key_cannot_be_duplicated_in_role_bindings():
    with pytest.raises(ValidationError, match="FlowPilot API keys must be unique"):
        Settings(
            deepseek_api_key="test-deepseek-key",
            flowpilot_auth_enabled=True,
            flowpilot_api_key="duplicate",
            flowpilot_api_keys=[{"key": "duplicate", "role": "operator"}],
            _env_file=None,
        )


def test_role_bound_keys_can_replace_legacy_key():
    settings = Settings(
        deepseek_api_key="test-deepseek-key",
        flowpilot_auth_enabled=True,
        flowpilot_api_key=None,
        flowpilot_api_keys=[{"key": "admin-list-key", "role": "admin"}],
        _env_file=None,
    )

    assert settings.flowpilot_api_key is None
    assert settings.flowpilot_api_keys[0].role.value == "admin"
