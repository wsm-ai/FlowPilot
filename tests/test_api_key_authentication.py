import hmac

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.agent import (
    get_persistent_agent_service,
    get_persistent_approval_workflow_service,
)
from app.core.config import Settings, get_settings
from app.main import app


TEST_API_KEY = "flowpilot-test-api-key"
AUTHORIZATION = {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def authenticated_app(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("FLOWPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("FLOWPILOT_API_KEY", TEST_API_KEY)
    get_settings.cache_clear()
    try:
        yield
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-key"},
        {"Authorization": "Basic flowpilot-test-api-key"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer first second"},
    ],
)
def test_protected_route_rejects_missing_or_invalid_credentials(
    authenticated_app,
    headers: dict[str, str],
):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers=headers,
            json={"message": ""},
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    assert response.headers["www-authenticate"] == "Bearer"
    assert TEST_API_KEY not in response.text


def test_bearer_scheme_is_case_insensitive_and_whitespace_is_accepted(
    authenticated_app,
):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers={"Authorization": f"bEaReR   {TEST_API_KEY}"},
            json={"message": ""},
        )

    assert response.status_code == 422


def test_correct_api_key_allows_protected_documentation(authenticated_app):
    with TestClient(app) as client:
        response = client.get("/openapi.json", headers=AUTHORIZATION)

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "FlowPilot API"


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_documentation_is_protected(authenticated_app, path: str):
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 401


def test_health_remains_public(authenticated_app):
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "FlowPilot"}


def test_query_string_api_key_is_not_accepted(authenticated_app):
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/chat?api_key={TEST_API_KEY}",
            json={"message": ""},
        )

    assert response.status_code == 401


def test_unauthorized_agent_request_does_not_execute_service(authenticated_app):
    calls = 0

    class ForbiddenService:
        async def run(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("service must not execute")

    app.dependency_overrides[get_persistent_agent_service] = ForbiddenService
    with TestClient(app) as client:
        response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 401
    assert calls == 0


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/v1/agent/plan-run", {"goal": "Create issue"}),
        (
            "/api/v1/agent/approval/resume",
            {
                "run_id": "run-1",
                "thread_id": "thread-1",
                "decision": "approve",
            },
        ),
        (
            "/api/v1/agent/answer/retry",
            {"run_id": "run-1", "thread_id": "thread-1"},
        ),
    ],
)
def test_unauthorized_planned_requests_do_not_execute_service(
    authenticated_app,
    path: str,
    payload: dict[str, str],
):
    calls = 0

    class ForbiddenService:
        async def start(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("service must not execute")

        async def resume(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("service must not execute")

        async def retry_grounded_answer(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("service must not execute")

    app.dependency_overrides[
        get_persistent_approval_workflow_service
    ] = ForbiddenService
    with TestClient(app) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 401
    assert calls == 0


def test_mounted_mcp_application_cannot_bypass_authentication(authenticated_app):
    with TestClient(app) as client:
        response = client.post("/mcp", json={})

    assert response.status_code == 401


def test_constant_time_comparison_is_used(authenticated_app, monkeypatch):
    calls: list[tuple[str, str]] = []
    original = hmac.compare_digest

    def recording_compare(left: str, right: str) -> bool:
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr(
        "app.security.api_key.hmac.compare_digest",
        recording_compare,
    )
    with TestClient(app) as client:
        response = client.get("/openapi.json", headers=AUTHORIZATION)

    assert response.status_code == 200
    assert calls == [(TEST_API_KEY, TEST_API_KEY)]


@pytest.mark.parametrize("configured_key", [None, "", "   "])
def test_enabled_authentication_requires_non_blank_server_key(
    monkeypatch,
    configured_key: str | None,
):
    monkeypatch.delenv("FLOWPILOT_API_KEY", raising=False)
    kwargs = {
        "deepseek_api_key": "test-deepseek-key",
        "flowpilot_auth_enabled": True,
        "_env_file": None,
    }
    if configured_key is not None:
        kwargs["flowpilot_api_key"] = configured_key

    with pytest.raises(
        ValidationError,
        match="FLOWPILOT_API_KEY is required when authentication is enabled",
    ):
        Settings(**kwargs)


def test_explicit_test_only_authentication_disable_allows_missing_key(monkeypatch):
    monkeypatch.delenv("FLOWPILOT_API_KEY", raising=False)

    settings = Settings(
        deepseek_api_key="test-deepseek-key",
        flowpilot_auth_enabled=False,
        _env_file=None,
    )

    assert settings.flowpilot_auth_enabled is False
    assert settings.flowpilot_api_key is None


def test_production_app_startup_fails_when_enabled_key_is_blank(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("FLOWPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("FLOWPILOT_API_KEY", "   ")
    get_settings.cache_clear()
    try:
        with pytest.raises(
            ValidationError,
            match=(
                "FLOWPILOT_API_KEY is required when authentication is enabled"
            ),
        ):
            with TestClient(app):
                pass
    finally:
        get_settings.cache_clear()
