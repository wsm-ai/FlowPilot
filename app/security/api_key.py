from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.security.rbac import ROLE_PERMISSIONS, Permission, Role, required_permission
from app.security.tool_authorization import (
    reset_tool_execution_role,
    set_tool_execution_role,
)

if TYPE_CHECKING:
    from app.core.config import Settings


SettingsProvider = Callable[[], "Settings"]


class APIKeyAuthenticationMiddleware:
    """Fail-closed API key authentication for FlowPilot HTTP requests."""

    def __init__(
        self,
        app: ASGIApp,
        settings_provider: SettingsProvider,
    ) -> None:
        self._app = app
        self._settings_provider = settings_provider

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if scope.get("method") == "GET" and scope.get("path") == "/health":
            await self._app(scope, receive, send)
            return

        settings = self._settings_provider()
        if not settings.flowpilot_auth_enabled:
            await self._app(scope, receive, send)
            return

        supplied_token = self._bearer_token(scope)
        role = self._authenticated_role(settings, supplied_token)
        if role is None:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        permission = required_permission(
            scope.get("method", ""),
            scope.get("path", ""),
        )
        if permission is not None and permission not in ROLE_PERMISSIONS[role]:
            response = JSONResponse(
                status_code=403,
                content={"detail": "Insufficient permissions"},
            )
            await response(scope, receive, send)
            return

        scope.setdefault("state", {})["flowpilot_role"] = role.value

        identity_token = set_tool_execution_role(role)
        try:
            await self._app(scope, receive, send)
        finally:
            reset_tool_execution_role(identity_token)

    @staticmethod
    def _authenticated_role(
        settings: Settings,
        supplied_token: str | None,
    ) -> Role | None:
        if supplied_token is None:
            return None
        matched_role: Role | None = None
        if settings.flowpilot_api_key is not None:
            if hmac.compare_digest(
                supplied_token,
                settings.flowpilot_api_key.get_secret_value(),
            ):
                matched_role = Role.ADMIN
        for binding in settings.flowpilot_api_keys:
            if hmac.compare_digest(
                supplied_token,
                binding.key.get_secret_value(),
            ):
                matched_role = binding.role
        return matched_role

    @staticmethod
    def _bearer_token(scope: Scope) -> str | None:
        authorization_values = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"authorization"
        ]
        if len(authorization_values) != 1:
            return None
        try:
            value = authorization_values[0].decode("latin-1")
        except UnicodeDecodeError:
            return None
        parts = value.strip().split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return parts[1]
