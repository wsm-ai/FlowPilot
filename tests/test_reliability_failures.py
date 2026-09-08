import asyncio
from dataclasses import FrozenInstanceError

import pytest

from app.mcp.client import (
    MCPConnectionError,
    MCPDiscoveryError,
    MCPToolCallError,
    MCPToolRegistrationError,
)
from app.persistence.repository import PersistenceError
from app.providers.base import LLMProviderError
from app.reliability.failures import (
    FailureCategory,
    FailureDescriptor,
    FailureDomain,
    classify_failure,
)
from app.services.grounded_answer_service import GroundedAnswerError
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError


@pytest.mark.parametrize(
    ("exception", "domain", "category", "code"),
    [
        (
            LLMProviderError("private"),
            FailureDomain.PROVIDER,
            FailureCategory.TRANSIENT,
            "provider_failure",
        ),
        (
            ToolExecutionError("private"),
            FailureDomain.TOOL,
            FailureCategory.PERMANENT,
            "tool_execution_failure",
        ),
        (
            PlanningError("private"),
            FailureDomain.PLANNING,
            FailureCategory.VALIDATION,
            "planning_failure",
        ),
        (
            MCPConnectionError("private"),
            FailureDomain.MCP,
            FailureCategory.TRANSIENT,
            "mcp_connection_failure",
        ),
        (
            MCPDiscoveryError("private"),
            FailureDomain.MCP,
            FailureCategory.TRANSIENT,
            "mcp_discovery_failure",
        ),
        (
            MCPToolCallError("private"),
            FailureDomain.MCP,
            FailureCategory.TRANSIENT,
            "mcp_tool_call_failure",
        ),
        (
            MCPToolRegistrationError("private"),
            FailureDomain.MCP,
            FailureCategory.CONFIGURATION,
            "mcp_tool_registration_failure",
        ),
        (
            GroundedAnswerError("private"),
            FailureDomain.GROUNDING,
            FailureCategory.TRANSIENT,
            "grounding_failure",
        ),
        (
            PersistenceError("private"),
            FailureDomain.PERSISTENCE,
            FailureCategory.TRANSIENT,
            "persistence_failure",
        ),
    ],
)
def test_known_failures_have_stable_classifications(
    exception, domain, category, code
):
    descriptor = classify_failure(exception)

    assert descriptor.domain is domain
    assert descriptor.category is category
    assert descriptor.code == code
    assert "private" not in descriptor.safe_message


def test_unknown_failure_does_not_expose_exception_message():
    descriptor = classify_failure(RuntimeError("PASSWORD=super-secret"))

    assert descriptor == FailureDescriptor(
        domain=FailureDomain.WORKFLOW,
        category=FailureCategory.PERMANENT,
        code="unclassified_failure",
        safe_message="Unexpected application failure",
    )
    assert "super-secret" not in descriptor.safe_message


def test_category_can_be_explicitly_overridden_for_side_effect_semantics():
    descriptor = classify_failure(
        MCPToolCallError("private remote state"),
        category=FailureCategory.AMBIGUOUS_SIDE_EFFECT,
    )

    assert descriptor.domain is FailureDomain.MCP
    assert descriptor.category is FailureCategory.AMBIGUOUS_SIDE_EFFECT
    assert descriptor.code == "mcp_tool_call_failure"


def test_domain_can_be_explicitly_overridden():
    descriptor = classify_failure(
        RuntimeError("private"),
        domain=FailureDomain.CONFIGURATION,
        category=FailureCategory.CONFIGURATION,
    )

    assert descriptor.domain is FailureDomain.CONFIGURATION
    assert descriptor.category is FailureCategory.CONFIGURATION
    assert descriptor.code == "unclassified_failure"


def test_failure_descriptor_is_immutable():
    descriptor = classify_failure(PlanningError("private"))

    with pytest.raises(FrozenInstanceError):
        descriptor.code = "changed"


def test_cancelled_error_is_classified_as_control_flow():
    descriptor = classify_failure(asyncio.CancelledError())

    assert descriptor.domain is FailureDomain.WORKFLOW
    assert descriptor.category is FailureCategory.CANCELLED
    assert descriptor.code == "operation_cancelled"


def test_classification_does_not_modify_original_exception():
    exception = MCPConnectionError("TOKEN=private")
    original_args = exception.args

    classify_failure(exception)

    assert exception.args == original_args


def test_classification_is_deterministic_and_does_not_parse_messages():
    first = classify_failure(MCPConnectionError("timeout"))
    second = classify_failure(MCPConnectionError("credential missing"))

    assert first == second
