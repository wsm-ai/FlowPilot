import asyncio
from dataclasses import dataclass
from enum import StrEnum

from app.persistence.repository import PersistenceError
from app.providers.base import LLMProviderError
from app.services.grounded_answer_service import GroundedAnswerError
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError
from app.reliability.side_effects import (
    SideEffectConflictError,
    SideEffectReplayBlockedError,
)
from app.reliability.timeouts import (
    ReadOperationTimeoutError,
    SideEffectOperationTimeoutError,
    UnknownOperationTimeoutError,
)


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
        ReadOperationTimeoutError,
        FailureDescriptor(
            domain=FailureDomain.WORKFLOW,
            category=FailureCategory.TRANSIENT,
            code="read_operation_timeout",
            safe_message="Read operation timed out",
        ),
    ),
    (
        SideEffectOperationTimeoutError,
        FailureDescriptor(
            domain=FailureDomain.WORKFLOW,
            category=FailureCategory.AMBIGUOUS_SIDE_EFFECT,
            code="side_effect_operation_timeout",
            safe_message="Side-effect operation timed out",
        ),
    ),
    (
        UnknownOperationTimeoutError,
        FailureDescriptor(
            domain=FailureDomain.WORKFLOW,
            category=FailureCategory.AMBIGUOUS_SIDE_EFFECT,
            code="unknown_operation_timeout",
            safe_message="Operation timed out",
        ),
    ),
    (
        SideEffectReplayBlockedError,
        FailureDescriptor(
            domain=FailureDomain.WORKFLOW,
            category=FailureCategory.AMBIGUOUS_SIDE_EFFECT,
            code="side_effect_replay_blocked",
            safe_message="Side-effect execution replay is blocked",
        ),
    ),
    (
        SideEffectConflictError,
        FailureDescriptor(
            domain=FailureDomain.WORKFLOW,
            category=FailureCategory.PERMANENT,
            code="side_effect_execution_conflict",
            safe_message="Side-effect execution conflicts with existing record",
        ),
    ),
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
    from app.mcp.client import (
        MCPAuthenticationError,
        MCPConnectionError,
        MCPConnectionConfigurationError,
        MCPDiscoveryError,
        MCPDiscoveryValidationError,
        MCPPermanentConnectionError,
        MCPPermanentDiscoveryError,
        MCPTransientConnectionError,
        MCPTransientDiscoveryError,
        MCPToolCallError,
        MCPToolRegistrationError,
    )

    mcp_mappings = (
        (
            MCPTransientConnectionError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.TRANSIENT,
                "mcp_transient_connection_failure",
                "MCP connection temporarily unavailable",
            ),
        ),
        (
            MCPAuthenticationError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.CONFIGURATION,
                "mcp_authentication_failure",
                "MCP authentication configuration failed",
            ),
        ),
        (
            MCPConnectionConfigurationError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.CONFIGURATION,
                "mcp_connection_configuration_failure",
                "MCP connection configuration failed",
            ),
        ),
        (
            MCPPermanentConnectionError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.PERMANENT,
                "mcp_permanent_connection_failure",
                "MCP protocol connection failed",
            ),
        ),
        (
            MCPConnectionError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.PERMANENT,
                "mcp_connection_failure",
                "MCP connection failed",
            ),
        ),
        (
            MCPTransientDiscoveryError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.TRANSIENT,
                "mcp_transient_discovery_failure",
                "MCP tool discovery temporarily unavailable",
            ),
        ),
        (
            MCPDiscoveryValidationError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.VALIDATION,
                "mcp_discovery_validation_failure",
                "MCP tool discovery returned invalid data",
            ),
        ),
        (
            MCPPermanentDiscoveryError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.PERMANENT,
                "mcp_permanent_discovery_failure",
                "MCP tool discovery failed",
            ),
        ),
        (
            MCPDiscoveryError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.PERMANENT,
                "mcp_discovery_failure",
                "MCP tool discovery failed",
            ),
        ),
        (
            MCPToolCallError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.TRANSIENT,
                "mcp_tool_call_failure",
                "MCP tool call failed",
            ),
        ),
        (
            MCPToolRegistrationError,
            FailureDescriptor(
                FailureDomain.MCP,
                FailureCategory.CONFIGURATION,
                "mcp_tool_registration_failure",
                "MCP tool registration failed",
            ),
        ),
    )
    descriptor = next(
        (
            candidate
            for exception_type, candidate in mcp_mappings + _FAILURE_MAPPINGS
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
