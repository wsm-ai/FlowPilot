from enum import Enum

from pydantic import BaseModel, SecretStr, field_validator


class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class Permission(str, Enum):
    AGENT_EXECUTE = "agent:execute"
    WORKFLOW_EXECUTE = "workflow:execute"
    APPROVAL_EXECUTE = "approval:execute"
    MCP_ACCESS = "mcp:access"
    ADMIN_ACCESS = "admin:access"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ADMIN: frozenset(Permission),
    Role.OPERATOR: frozenset(
        {
            Permission.AGENT_EXECUTE,
            Permission.WORKFLOW_EXECUTE,
        }
    ),
    # FlowPilot currently has no read-only run or trace HTTP endpoint. Keep the
    # viewer role fail-closed until such an endpoint has an explicit mapping.
    Role.VIEWER: frozenset(),
}


class APIKeyRoleBinding(BaseModel):
    key: SecretStr
    role: Role

    @field_validator("key")
    @classmethod
    def normalize_key(cls, value: SecretStr) -> SecretStr:
        normalized = value.get_secret_value().strip()
        if not normalized:
            raise ValueError("RBAC API keys must not be blank")
        return SecretStr(normalized)


def required_permission(method: str, path: str) -> Permission | None:
    """Return the explicit permission for a protected production route."""
    route_permissions = {
        ("POST", "/api/v1/chat"): Permission.AGENT_EXECUTE,
        ("POST", "/api/v1/agent/run"): Permission.AGENT_EXECUTE,
        ("POST", "/api/v1/agent/plan-run"): Permission.WORKFLOW_EXECUTE,
        ("POST", "/api/v1/agent/approval/resume"): Permission.APPROVAL_EXECUTE,
        ("POST", "/api/v1/agent/answer/retry"): Permission.WORKFLOW_EXECUTE,
        ("GET", "/docs"): Permission.ADMIN_ACCESS,
        ("GET", "/docs/oauth2-redirect"): Permission.ADMIN_ACCESS,
        ("GET", "/redoc"): Permission.ADMIN_ACCESS,
        ("GET", "/openapi.json"): Permission.ADMIN_ACCESS,
    }
    permission = route_permissions.get((method, path))
    if permission is not None:
        return permission
    if path == "/mcp" or path.startswith("/mcp/"):
        return Permission.MCP_ACCESS
    # Every unclassified HTTP route is admin-only until explicitly reviewed.
    # The authentication middleware handles the sole public exception, /health,
    # before consulting this permission map.
    return Permission.ADMIN_ACCESS
