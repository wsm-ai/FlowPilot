from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

if TYPE_CHECKING:
    from app.core.config import Settings


SettingsProvider = Callable[[], "Settings"]
MAX_CONTENT_LENGTH_DIGITS = 20
MAX_BUFFERED_BODY_MESSAGES = 1024


class RequestBodyLimitMiddleware:
    """Bound HTTP request bodies before application-level body parsing."""

    def __init__(self, app: ASGIApp, settings_provider: SettingsProvider) -> None:
        self._app = app
        self._settings_provider = settings_provider

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or (
            scope.get("method") == "GET" and scope.get("path") == "/health"
        ):
            await self._app(scope, receive, send)
            return

        limit = self._settings_provider().flowpilot_max_request_body_bytes
        content_lengths = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"content-length"
        ]
        if len(content_lengths) > 1:
            await self._invalid_content_length(scope, receive, send)
            return
        if content_lengths:
            try:
                raw_content_length = content_lengths[0].decode("ascii")
            except UnicodeDecodeError:
                await self._invalid_content_length(scope, receive, send)
                return
            if not raw_content_length.isdigit():
                await self._invalid_content_length(scope, receive, send)
                return
            significant_length = raw_content_length.lstrip("0") or "0"
            limit_text = str(limit)
            if (
                len(significant_length) > len(limit_text)
                or (
                    len(significant_length) == len(limit_text)
                    and significant_length > limit_text
                )
            ):
                await self._payload_too_large(scope, receive, send)
                return
            if len(raw_content_length) > MAX_CONTENT_LENGTH_DIGITS:
                await self._invalid_content_length(scope, receive, send)
                return

        buffered: deque[Message] = deque()
        actual_length = 0
        body_message_count = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            body_message_count += 1
            if body_message_count > MAX_BUFFERED_BODY_MESSAGES:
                await self._payload_too_large(scope, receive, send)
                return
            actual_length += len(message.get("body", b""))
            if actual_length > limit:
                await self._payload_too_large(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if buffered:
                return buffered.popleft()
            return await receive()

        await self._app(scope, replay_receive, send)

    @staticmethod
    async def _payload_too_large(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = JSONResponse(
            status_code=413,
            content={"detail": "Request body too large"},
        )
        await response(scope, receive, send)

    @staticmethod
    async def _invalid_content_length(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = JSONResponse(
            status_code=400,
            content={"detail": "Invalid Content-Length"},
        )
        await response(scope, receive, send)


class SecurityHeadersMiddleware:
    """Add API-safe response headers without changing application payloads."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_with_security_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {b"x-content-type-options", b"referrer-policy"}
                if scope.get("path") != "/health":
                    names.add(b"cache-control")
                headers = [
                    (name, value)
                    for name, value in headers
                    if name.lower() not in names
                ]
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                    ]
                )
                if scope.get("path") != "/health":
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, send_with_security_headers)
