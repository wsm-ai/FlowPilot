from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from app.security.rbac import Role
from app.reliability.side_effects import SideEffectExecutionError, digest_arguments


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    SIDE_EFFECT = "side_effect"
    HIGH_RISK = "high_risk"
    UNCLASSIFIED = "unclassified"


class ToolAuthorizationError(Exception):
    """Raised when a tool execution is not authorized."""


@dataclass(frozen=True, slots=True)
class ToolExecutionIdentity:
    role: Role | None


@dataclass(frozen=True, slots=True)
class ApprovedToolExecutionGrant:
    run_id: str
    thread_id: str
    step_id: int
    action: str
    arguments_digest: str
    _authority: object = field(repr=False, compare=False)


class ApprovalGrantAuthority:
    """Process-local capability used by a trusted approval service."""

    def __init__(self) -> None:
        self._seal = object()

    def issue(
        self,
        *,
        run_id: str,
        thread_id: str,
        step_id: int,
        action: str,
        arguments: dict,
    ) -> ApprovedToolExecutionGrant:
        return ApprovedToolExecutionGrant(
            run_id=run_id,
            thread_id=thread_id,
            step_id=step_id,
            action=action,
            arguments_digest=digest_arguments(arguments),
            _authority=self._seal,
        )

    @contextmanager
    def activate(self, grant: ApprovedToolExecutionGrant) -> Iterator[None]:
        if grant._authority is not self._seal:
            raise ToolAuthorizationError("Approval grant is not trusted")
        token = _approval_grant.set(grant)
        try:
            yield
        finally:
            _approval_grant.reset(token)


_identity: ContextVar[ToolExecutionIdentity | None] = ContextVar(
    "flowpilot_tool_execution_identity", default=None
)
_approval_grant: ContextVar[ApprovedToolExecutionGrant | None] = ContextVar(
    "flowpilot_approved_tool_execution_grant", default=None
)


def set_tool_execution_role(role: Role) -> Token[ToolExecutionIdentity | None]:
    return _identity.set(ToolExecutionIdentity(role=role))


def reset_tool_execution_role(token: Token[ToolExecutionIdentity | None]) -> None:
    _identity.reset(token)


@contextmanager
def tool_execution_role(role: Role) -> Iterator[None]:
    """Set a trusted role for a controlled internal execution scope."""
    token = set_tool_execution_role(role)
    try:
        yield
    finally:
        reset_tool_execution_role(token)


def current_approval_grant() -> ApprovedToolExecutionGrant | None:
    return _approval_grant.get()


def authorize_tool_execution(
    name: str,
    risk: ToolRisk,
    arguments: dict[str, Any] | None = None,
) -> None:
    identity = _identity.get()
    role = None if identity is None else identity.role

    if risk is ToolRisk.UNCLASSIFIED:
        raise ToolAuthorizationError("Tool execution is not authorized")

    if risk is ToolRisk.READ_ONLY:
        if role is None or role in {Role.ADMIN, Role.OPERATOR}:
            return
        raise ToolAuthorizationError("Tool execution is not authorized")

    grant = _approval_grant.get()
    if grant is None or grant.action != name:
        raise ToolAuthorizationError("Tool approval is required")
    if arguments is None:
        raise ToolAuthorizationError("Tool approval does not match arguments")
    try:
        arguments_digest = digest_arguments(arguments)
    except SideEffectExecutionError as exc:
        raise ToolAuthorizationError(
            "Tool approval does not match arguments"
        ) from exc
    if grant.arguments_digest != arguments_digest:
        raise ToolAuthorizationError("Tool approval does not match arguments")

    # A grant is installed only by ApprovalWorkflowService after validating a
    # durable checkpoint. This is the explicit internal authorization source
    # for non-HTTP service execution. Authenticated requests must be admin.
    if role is None or role is Role.ADMIN:
        return
    raise ToolAuthorizationError("Tool execution is not authorized")
