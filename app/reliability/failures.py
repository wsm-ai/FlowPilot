import asyncio
from dataclasses import dataclass
from enum import StrEnum

from app.mcp.client import (
    MCPConnectionError,
    MCPDiscoveryError,
    MCPToolCallError,
    MCPToolRegistrationError,
)
from app.persistence.repository import PersistenceError
from app.providers.base import LLMProviderError
from app.services.grounded_answer_service import GroundedAnswerError
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError


class FailureCategory(StrEnum):
    TRANSIENT = "transient"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    PERMANENT = "permanent"
    AMBIGUOUS_SIDE_EFFECT = "ambiguous_side_effect"
    BUSINESS_TERMINAL = "business_terminal"
    CANCELLED = "cancelled"


class FailureDomain(StrEnum):
    PROVIDER = "provider"
    TOOL = "tool"
    MCP = "mcp"
    PLANNING = "planning"
    PERSISTENCE = "persistence"
    GROUNDING = "grounding"
    WORKFLOW = "workflow"
    CONFIGURATION = "configuration"


@dataclass(frozen=True, slots=True)
class FailureDescriptor:
    domain: FailureDomain
    category: FailureCategory
    code: str
    safe_message: str


_FAILURE_MAPPINGS: tuple[
    tuple[type[BaseException], FailureDescriptor], ...
] = (
    (
        asyncio.CancelledError,
        FailureDescriptor(
            domain=FailureDomain.WORKFLOW,
            category=FailureCategory.CANCELLED,
            code="operation_cancelled",
            safe_message="Operation cancelled",
        ),
    ),
    (
        MCPConnectionError,
        FailureDescriptor(
            domain=FailureDomain.MCP,
            category=FailureCategory.TRANSIENT,
            code="mcp_connection_failure",
            safe_message="MCP connection failed",
        ),
    ),
    (
        MCPDiscoveryError,
        FailureDescriptor(
            domain=FailureDomain.MCP,
            category=FailureCategory.TRANSIENT,
            code="mcp_discovery_failure",
            safe_message="MCP tool discovery failed",
        ),
    ),
    (
        MCPToolCallError,
        FailureDescriptor(
            domain=FailureDomain.MCP,
            category=FailureCategory.TRANSIENT,
            code="mcp_tool_call_failure",
            safe_message="MCP tool call failed",
        ),
    ),
    (
        MCPToolRegistrationError,
        FailureDescriptor(
            domain=FailureDomain.MCP,
            category=FailureCategory.CONFIGURATION,
            code="mcp_tool_registration_failure",
            safe_message="MCP tool registration failed",
        ),
    ),
    (
        LLMProviderError,
        FailureDescriptor(
            domain=FailureDomain.PROVIDER,
            category=FailureCategory.TRANSIENT,
            code="provider_failure",
            safe_message="Language model provider failure",
        ),
    ),
    (
        ToolExecutionError,
        FailureDescriptor(
            domain=FailureDomain.TOOL,
            category=FailureCategory.PERMANENT,
            code="tool_execution_failure",
            safe_message="Tool execution failed",
        ),
    ),
    (
        PlanningError,
        FailureDescriptor(
            domain=FailureDomain.PLANNING,
            category=FailureCategory.VALIDATION,
            code="planning_failure",
            safe_message="Planning validation failed",
        ),
    ),
    (
        PersistenceError,
        FailureDescriptor(
            domain=FailureDomain.PERSISTENCE,
            category=FailureCategory.TRANSIENT,
            code="persistence_failure",
            safe_message="Persistence operation failed",
        ),
    ),
    (
        GroundedAnswerError,
        FailureDescriptor(
            domain=FailureDomain.GROUNDING,
            category=FailureCategory.TRANSIENT,
            code="grounding_failure",
            safe_message="Grounded answer synthesis failed",
        ),
    ),
)


_UNCLASSIFIED_FAILURE = FailureDescriptor(
    domain=FailureDomain.WORKFLOW,
    category=FailureCategory.PERMANENT,
    code="unclassified_failure",
    safe_message="Unexpected application failure",
)


def classify_failure(
    exc: BaseException,
    *,
    domain: FailureDomain | None = None,
    category: FailureCategory | None = None,
) -> FailureDescriptor:
    """Classify an exception without exposing or inspecting its message."""
    descriptor = next(
        (
            candidate
            for exception_type, candidate in _FAILURE_MAPPINGS
            if isinstance(exc, exception_type)
        ),
        _UNCLASSIFIED_FAILURE,
    )
    return FailureDescriptor(
        domain=domain if domain is not None else descriptor.domain,
        category=category if category is not None else descriptor.category,
        code=descriptor.code,
        safe_message=descriptor.safe_message,
    )
