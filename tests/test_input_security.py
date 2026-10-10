import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.agent import (
    get_persistent_agent_service,
    get_persistent_approval_workflow_service,
)
from app.api.dependencies import get_llm_service
from app.core.config import Settings, get_settings
from app.main import app
from app.security.http import RequestBodyLimitMiddleware


ADMIN_KEY = "input-security-admin-key"
AUTHORIZATION = {"Authorization": f"Bearer {ADMIN_KEY}"}


@pytest.fixture
def secured_app(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("FLOWPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("FLOWPILOT_API_KEY", ADMIN_KEY)
    monkeypatch.setenv("FLOWPILOT_API_KEYS", "[]")
    monkeypatch.setenv("FLOWPILOT_MAX_REQUEST_BODY_BYTES", "16384")
    get_settings.cache_clear()
    try:
        yield
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()


def test_valid_multilingual_and_multiline_chat_reaches_service(secured_app):
    received: list[str] = []

    class FakeLLMService:
        model = "fake-model"

        async def chat(self, message: str) -> str:
            received.append(message)
            return "ok"

    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    message = "请分析：\n```python\nprint('合法内容')\n```"
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers=AUTHORIZATION,
            json={"message": message},
        )

    assert response.status_code == 200
    assert received == [message]


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": 123}, {"message": ""}, {"message": "   "}],
)
def test_invalid_chat_input_returns_422_without_service_call(
    secured_app,
    payload,
):
    calls = 0

    class ForbiddenLLMService:
        model = "fake-model"

        async def chat(self, message: str) -> str:
            nonlocal calls
            calls += 1
            raise AssertionError("LLM service must not execute")

    app.dependency_overrides[get_llm_service] = lambda: ForbiddenLLMService()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers=AUTHORIZATION,
            json=payload,
        )

    assert response.status_code == 422
    assert calls == 0


def test_overlong_message_and_goal_return_422(secured_app):
    with TestClient(app) as client:
        message_response = client.post(
            "/api/v1/chat",
            headers=AUTHORIZATION,
            json={"message": "x" * 10_001},
        )
        goal_response = client.post(
            "/api/v1/agent/plan-run",
            headers=AUTHORIZATION,
            json={"goal": "x" * 10_001},
        )

    assert message_response.status_code == 422
    assert goal_response.status_code == 422


def test_schema_length_limits_are_independent_of_http_body_limit():
    from app.schemas.chat import ChatRequest
    from app.schemas.planned_agent import PlannedAgentRunRequest

    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 10_001)
    with pytest.raises(ValidationError):
        PlannedAgentRunRequest(goal="x" * 10_001)


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/api/v1/agent/run",
            {"message": "hello", "thread_id": "t" * 201},
        ),
        (
            "/api/v1/agent/approval/resume",
            {"run_id": "r" * 201, "thread_id": "thread", "decision": "approve"},
        ),
        (
            "/api/v1/agent/answer/retry",
            {"run_id": "run", "thread_id": "t" * 201},
        ),
    ],
)
def test_overlong_identifiers_return_422_without_business_execution(
    secured_app,
    path,
    payload,
):
    calls = 0

    class ForbiddenService:
        async def run(self, *args, **kwargs):
            nonlocal calls
            calls += 1

        async def resume(self, *args, **kwargs):
            nonlocal calls
            calls += 1

        async def retry_grounded_answer(self, *args, **kwargs):
            nonlocal calls
            calls += 1

    app.dependency_overrides[get_persistent_agent_service] = lambda: ForbiddenService()
    app.dependency_overrides[
        get_persistent_approval_workflow_service
    ] = lambda: ForbiddenService()
    with TestClient(app) as client:
        response = client.post(path, headers=AUTHORIZATION, json=payload)

    assert response.status_code == 422
    assert calls == 0


def test_invalid_approval_decision_and_extra_role_are_rejected(secured_app):
    with TestClient(app) as client:
        decision_response = client.post(
            "/api/v1/agent/approval/resume",
            headers=AUTHORIZATION,
            json={
                "run_id": "run",
                "thread_id": "thread",
                "decision": "escalate",
            },
        )
        role_response = client.post(
            "/api/v1/agent/run",
            headers=AUTHORIZATION,
            json={"message": "hello", "role": "admin"},
        )

    assert decision_response.status_code == 422
    assert role_response.status_code == 422


def test_oversized_json_returns_safe_413(secured_app):
    sensitive = "TOKEN=must-not-be-reflected"
    body = '{"message":"' + sensitive + ("x" * 20_000) + '"}'
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            content=body,
            headers={**AUTHORIZATION, "Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body too large"}
    assert sensitive not in response.text


async def _invoke_body_limiter(
    chunks: list[bytes],
    *,
    limit: int,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[list[bytes], list[dict]]:
    received_by_app: list[bytes] = []
    sent: list[dict] = []
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        sent.append(message)

    async def downstream(scope, downstream_receive, downstream_send):
        while True:
            message = await downstream_receive()
            received_by_app.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await downstream_send(
            {"type": "http.response.start", "status": 204, "headers": []}
        )
        await downstream_send({"type": "http.response.body", "body": b""})

    middleware = RequestBodyLimitMiddleware(
        downstream,
        settings_provider=lambda: SimpleNamespace(
            flowpilot_max_request_body_bytes=limit
        ),
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/test",
        "headers": headers or [],
    }
    await middleware(scope, receive, send)
    return received_by_app, sent


def test_missing_content_length_and_chunking_cannot_bypass_limit():
    received, sent = asyncio.run(
        _invoke_body_limiter([b"1234", b"5678"], limit=7)
    )

    assert received == []
    assert sent[0]["status"] == 413


def test_false_small_content_length_cannot_bypass_actual_size_check():
    received, sent = asyncio.run(
        _invoke_body_limiter(
            [b"1234", b"5678"],
            limit=7,
            headers=[(b"content-length", b"1")],
        )
    )

    assert received == []
    assert sent[0]["status"] == 413


def test_valid_chunked_body_is_replayed_unchanged():
    received, sent = asyncio.run(
        _invoke_body_limiter([b"1234", b"56"], limit=6)
    )

    assert received == [b"1234", b"56"]
    assert sent[0]["status"] == 204


def test_malformed_content_length_is_rejected_safely():
    received, sent = asyncio.run(
        _invoke_body_limiter(
            [b"body"],
            limit=10,
            headers=[(b"content-length", b"+4")],
        )
    )

    assert received == []
    assert sent[0]["status"] == 400


def test_extremely_long_numeric_content_length_returns_413_not_500():
    received, sent = asyncio.run(
        _invoke_body_limiter(
            [b"body"],
            limit=10,
            headers=[(b"content-length", b"9" * 10_000)],
        )
    )

    assert received == []
    assert sent[0]["status"] == 413


def test_excessive_leading_zero_content_length_is_rejected_as_malformed():
    received, sent = asyncio.run(
        _invoke_body_limiter(
            [b"body"],
            limit=10,
            headers=[(b"content-length", (b"0" * 10_000) + b"4")],
        )
    )

    assert received == []
    assert sent[0]["status"] == 400


def test_excessive_empty_body_chunks_are_bounded_before_application():
    received, sent = asyncio.run(
        _invoke_body_limiter([b""] * 1025, limit=10)
    )

    assert received == []
    assert sent[0]["status"] == 413


def test_many_normal_chunks_are_replayed_in_original_order():
    chunks = [bytes([index % 251]) for index in range(1024)]
    received, sent = asyncio.run(_invoke_body_limiter(chunks, limit=1024))

    assert received == chunks
    assert sent[0]["status"] == 204


def test_unauthenticated_oversized_request_still_returns_401(secured_app):
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            content="x" * 2000,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 401


def test_security_headers_cover_success_and_security_errors(secured_app):
    with TestClient(app) as client:
        health = client.get("/health")
        unauthorized = client.post("/api/v1/chat", json={"message": "hello"})

    for response in (health, unauthorized):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
    assert unauthorized.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("value", [0, -1, 16_777_217])
def test_request_body_limit_configuration_is_bounded(value):
    with pytest.raises(ValidationError):
        Settings(
            deepseek_api_key="test-key",
            flowpilot_auth_enabled=False,
            flowpilot_max_request_body_bytes=value,
            _env_file=None,
        )
