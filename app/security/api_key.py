import hmac
from collections.abc import Callable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import Settings


SettingsProvider = Callable[[], Settings]


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
        configured_key = settings.flowpilot_api_key
        authenticated = (
            supplied_token is not None
            and configured_key is not None
            and hmac.compare_digest(
                supplied_token,
                configured_key.get_secret_value(),
            )
        )
        if not authenticated:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)

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
