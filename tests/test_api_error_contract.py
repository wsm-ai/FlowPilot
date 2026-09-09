import asyncio
from dataclasses import FrozenInstanceError

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.errors import (
    APIErrorResponse,
    http_status_for_failure,
    register_api_exception_handlers,
    response_for_failure,
)
from app.mcp.client import (
    MCPAuthenticationError,
    MCPConnectionConfigurationError,
    MCPDiscoveryValidationError,
    MCPPermanentConnectionError,
    MCPTransientConnectionError,
)
from app.reliability.failures import (
    FailureCategory,
    FailureDescriptor,
    FailureDomain,
)
from app.reliability.side_effects import SideEffectReplayBlockedError
from app.reliability.timeouts import (
    ReadOperationTimeoutError,
    SideEffectOperationTimeoutError,
)
from app.services.planner_service import PlanningError


def descriptor(category):
    return FailureDescriptor(
        domain=FailureDomain.WORKFLOW,
        category=category,
        code="stable_code",
        safe_message="Safe message",
    )


@pytest.mark.parametrize(
    ("category", "status_code"),
    [
        (FailureCategory.VALIDATION, 422),
        (FailureCategory.BUSINESS_TERMINAL, 409),
        (FailureCategory.AMBIGUOUS_SIDE_EFFECT, 409),
        (FailureCategory.TRANSIENT, 503),
        (FailureCategory.CONFIGURATION, 500),
        (FailureCategory.PERMANENT, 500),
        (FailureCategory.CANCELLED, 500),
    ],
)
def test_failure_descriptor_has_deterministic_http_status(category, status_code):
    assert http_status_for_failure(descriptor(category)) == status_code


def test_response_schema_is_typed_stable_and_immutable():
    response = response_for_failure(descriptor(FailureCategory.TRANSIENT))
    assert isinstance(response, APIErrorResponse)
    assert response.model_dump(mode="json") == {
        "error": {
            "code": "stable_code",
            "message": "Safe message",
            "category": "transient",
            "domain": "workflow",
        }
    }
    with pytest.raises((FrozenInstanceError, ValueError)):
        response.error.code = "changed"


def client_for_exception(error):
    app = FastAPI()
    register_api_exception_handlers(app)

    @app.get("/failure")
    async def failure_endpoint():
        raise error

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (ReadOperationTimeoutError("private"), 503, "read_operation_timeout"),
        (
            SideEffectOperationTimeoutError("private"),
            409,
            "side_effect_operation_timeout",
        ),
        (
            MCPTransientConnectionError("private"),
            503,
            "mcp_transient_connection_failure",
        ),
        (MCPAuthenticationError("private"), 500, "mcp_authentication_failure"),
        (
            MCPConnectionConfigurationError("private"),
            500,
            "mcp_connection_configuration_failure",
        ),
        (
            MCPPermanentConnectionError("private"),
            500,
            "mcp_permanent_connection_failure",
        ),
        (
            MCPDiscoveryValidationError("private"),
            422,
            "mcp_discovery_validation_failure",
        ),
        (SideEffectReplayBlockedError("private"), 409,
         "side_effect_replay_blocked"),
        (PlanningError("private"), 422, "planning_failure"),
        (RuntimeError("private"), 500, "unclassified_failure"),
    ],
)
def test_known_and_unknown_application_failures_use_unified_contract(
    error, status_code, code
):
    with client_for_exception(error) as client:
        response = client.get("/failure")
    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert set(response.json()["error"]) == {
        "code", "message", "category", "domain"
    }
    assert "private" not in response.text


def test_raw_exception_secrets_are_not_returned():
    error = RuntimeError(
        "Bearer SECRET_TOKEN https://user:password@example.com"
    )
    with client_for_exception(error) as client:
        response = client.get("/failure")
    assert response.status_code == 500
    assert response.json()["error"]["message"] == (
        "Unexpected application failure"
    )
    assert "SECRET_TOKEN" not in response.text
    assert "password" not in response.text


def test_request_validation_uses_unified_safe_contract():
    class Payload(BaseModel):
        message: str

    app = FastAPI()
    register_api_exception_handlers(app)

    @app.post("/validate")
    async def validate(payload: Payload):
        return payload

    with TestClient(app) as client:
        response = client.post("/validate", json={"message": {"secret": "x"}})
    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "request_validation_error",
            "message": "Request validation failed",
            "category": "validation",
            "domain": "workflow",
        }
    }
    assert "secret" not in response.text


def test_framework_404_and_explicit_http_exception_are_preserved():
    app = FastAPI()
    register_api_exception_handlers(app)

    @app.get("/conflict")
    async def conflict():
        raise HTTPException(status_code=409, detail="Explicit conflict")

    with TestClient(app, raise_server_exceptions=False) as client:
        missing = client.get("/missing")
        conflict_response = client.get("/conflict")
    assert missing.status_code == 404
    assert conflict_response.status_code == 409
    assert conflict_response.json() == {"detail": "Explicit conflict"}


def test_ambiguous_side_effect_response_does_not_retry_dispatch():
    calls = 0
    app = FastAPI()
    register_api_exception_handlers(app)

    @app.post("/side-effect")
    async def side_effect():
        nonlocal calls
        calls += 1
        raise SideEffectOperationTimeoutError("TOKEN=private")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/side-effect")
    assert response.status_code == 409
    assert response.json()["error"]["category"] == "ambiguous_side_effect"
    assert calls == 1


def test_cancelled_error_is_not_converted_by_application_handler():
    assert not isinstance(asyncio.CancelledError(), Exception)
