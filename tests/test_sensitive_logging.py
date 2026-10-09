from io import StringIO
import logging

import pytest
from fastapi.testclient import TestClient
from uvicorn.logging import AccessFormatter

from app.api.dependencies import get_llm_service
from app.core.config import Settings, get_settings
from app.main import app
from app.security.rbac import APIKeyRoleBinding
from app.security.redaction import (
    REDACTED,
    SensitiveDataFilter,
    configure_sensitive_logging,
    redact_sensitive_data,
    redact_text,
)


FLOWPILOT_KEY = "flowpilot-super-secret"
OPERATOR_KEY = "operator-super-secret"
DEEPSEEK_KEY = "deepseek-super-secret"


def _capturing_logger(name: str, *, secrets=()) -> tuple[logging.Logger, StringIO]:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    handler.addFilter(SensitiveDataFilter(secrets=secrets))
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.filters.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, stream


def test_settings_repr_and_dump_do_not_reveal_api_keys():
    settings = Settings(
        deepseek_api_key=DEEPSEEK_KEY,
        flowpilot_auth_enabled=True,
        flowpilot_api_key=FLOWPILOT_KEY,
        flowpilot_api_keys=[{"key": OPERATOR_KEY, "role": "operator"}],
        _env_file=None,
    )

    rendered = repr(settings)
    dumped = repr(settings.model_dump())
    for secret in (DEEPSEEK_KEY, FLOWPILOT_KEY, OPERATOR_KEY):
        assert secret not in rendered
        assert secret not in dumped
    assert "**********" in rendered


def test_api_key_role_binding_repr_does_not_reveal_key():
    binding = APIKeyRoleBinding(key=OPERATOR_KEY, role="operator")

    assert OPERATOR_KEY not in repr(binding)
    assert "**********" in repr(binding)


def test_nested_structured_data_redacts_sensitive_keys_case_insensitively():
    payload = {
        "Authorization": "Bearer auth-secret",
        "nested": {
            "access_token": "access-secret",
            "PASSWORD": "password-secret",
            "client-secret": "client-secret-value",
        },
        "status_code": 503,
        "request_path": "/api/v1/chat",
        "monkey": "normal word containing key",
    }

    redacted = redact_sensitive_data(payload)

    assert redacted["Authorization"] == REDACTED
    assert redacted["nested"]["access_token"] == REDACTED
    assert redacted["nested"]["PASSWORD"] == REDACTED
    assert redacted["nested"]["client-secret"] == REDACTED
    assert redacted["status_code"] == 503
    assert redacted["request_path"] == "/api/v1/chat"
    assert redacted["monkey"] == "normal word containing key"


def test_bearer_assignments_and_url_credentials_are_redacted():
    rendered = redact_text(
        "Authorization=Bearer bearer-secret "
        "api_key=api-secret password:password-secret "
        "url=https://user:pass@example.invalid/path"
    )

    assert "bearer-secret" not in rendered
    assert "api-secret" not in rendered
    assert "password-secret" not in rendered
    assert "user:pass" not in rendered
    assert rendered.count(REDACTED) >= 4


def test_filter_redacts_configured_single_multi_role_and_deepseek_keys():
    secrets = (FLOWPILOT_KEY, OPERATOR_KEY, DEEPSEEK_KEY)
    logger, stream = _capturing_logger(
        "flowpilot.tests.sensitive.known",
        secrets=secrets,
    )

    logger.info(
        "configured values: %s %s %s",
        FLOWPILOT_KEY,
        OPERATOR_KEY,
        DEEPSEEK_KEY,
    )

    output = stream.getvalue()
    assert REDACTED in output
    assert all(secret not in output for secret in secrets)


def test_logger_exception_preserves_type_but_not_sensitive_message():
    logger, stream = _capturing_logger("flowpilot.tests.sensitive.exception")

    try:
        raise RuntimeError("TOKEN=exception-secret")
    except RuntimeError:
        logger.exception("Provider request failed for api_key=message-secret")

    output = stream.getvalue()
    assert "RuntimeError" in output
    assert "Provider request failed" in output
    assert "exception-secret" not in output
    assert "message-secret" not in output
    assert REDACTED in output


def test_configure_sensitive_logging_is_idempotent_and_does_not_touch_root():
    root = logging.getLogger()
    root_filters = tuple(root.filters)
    logger_name = "flowpilot.tests.sensitive.configured"
    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler(StringIO())
    logger.handlers = [handler]
    logger.filters.clear()

    configure_sensitive_logging(
        secrets=[FLOWPILOT_KEY],
        logger_names=[logger_name],
    )
    configure_sensitive_logging(
        secrets=[OPERATOR_KEY],
        logger_names=[logger_name],
    )

    protected = [
        item
        for item in handler.filters
        if isinstance(item, SensitiveDataFilter)
    ]
    assert len(protected) == 1
    assert tuple(root.filters) == root_filters


def test_child_logger_records_are_filtered_by_parent_handler():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s %(message)s"))
    parent_name = "flowpilot.tests.sensitive.parent"
    parent = logging.getLogger(parent_name)
    parent.handlers = [handler]
    parent.propagate = False
    parent.setLevel(logging.INFO)
    child = logging.getLogger(f"{parent_name}.child")
    child.handlers = []
    child.filters = []
    child.propagate = True
    child.setLevel(logging.INFO)
    configure_sensitive_logging(
        logger_names=[parent_name],
        secrets=[FLOWPILOT_KEY],
    )

    child.info("child received api_key=%s", FLOWPILOT_KEY)

    output = stream.getvalue()
    assert child.name in output
    assert FLOWPILOT_KEY not in output
    assert REDACTED in output


def test_parameterized_nested_sensitive_values_are_redacted_before_formatting():
    logger, stream = _capturing_logger("flowpilot.tests.sensitive.arguments")
    payload = {
        "profile": {
            "password": "private123",
            "items": [
                {"Access-Token": "nested-token"},
                {"label": "safe-value"},
            ],
        }
    }

    logger.info("用户配置: %s", payload)

    output = stream.getvalue()
    assert "private123" not in output
    assert "nested-token" not in output
    assert "safe-value" in output
    assert "用户配置" in output
    assert output.count(REDACTED) == 2


def test_uvicorn_access_formatter_keeps_required_argument_shape_and_redacts():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s'))
    handler.addFilter(SensitiveDataFilter())
    logger = logging.getLogger("uvicorn.access")
    original_handlers = logger.handlers[:]
    original_filters = logger.filters[:]
    original_propagate = logger.propagate
    original_level = logger.level
    logger.handlers = [handler]
    logger.filters = []
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:1234",
            "GET",
            "/internal?access_token=query-secret",
            "1.1",
            401,
        )
    finally:
        logger.handlers = original_handlers
        logger.filters = original_filters
        logger.propagate = original_propagate
        logger.setLevel(original_level)

    output = stream.getvalue()
    assert "query-secret" not in output
    assert REDACTED in output
    assert "GET" in output
    assert "401" in output


def test_application_exception_response_does_not_expose_secret(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY)
    monkeypatch.setenv("FLOWPILOT_AUTH_ENABLED", "true")
    monkeypatch.setenv("FLOWPILOT_API_KEY", FLOWPILOT_KEY)
    monkeypatch.setenv("FLOWPILOT_API_KEYS", "[]")
    get_settings.cache_clear()

    class FailingService:
        model = "fake-model"

        async def chat(self, message: str) -> str:
            raise RuntimeError(f"api_key={DEEPSEEK_KEY}")

    app.dependency_overrides[get_llm_service] = lambda: FailingService()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/chat",
                headers={"Authorization": f"Bearer {FLOWPILOT_KEY}"},
                json={"message": "hello"},
            )
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 500
    assert DEEPSEEK_KEY not in response.text
    assert FLOWPILOT_KEY not in response.text
    assert response.json()["error"]["message"] == "Unexpected application failure"
