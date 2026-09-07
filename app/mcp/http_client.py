from contextlib import AsyncExitStack
from typing import Any

import httpx2
from mcp import Client, MCPError as SDKMCPError
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr

from app.mcp.client import MCPConnectionError
from app.mcp.config import MCPHTTPServerConfig
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.sdk_client import (
    TRANSPORT_ERRORS,
    call_tool,
    discover_tools,
    is_sdk_timeout,
)


class StreamableHTTPMCPClient:
    def __init__(
        self,
        config: MCPHTTPServerConfig,
        *,
        bearer_token: SecretStr | None = None,
    ) -> None:
        self._config = config.model_copy(deep=True)
        self._bearer_token = bearer_token
        self._client: Client | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "StreamableHTTPMCPClient":
        if self._client is not None:
            raise MCPConnectionError("MCP client is already connected")

        stack = AsyncExitStack()
        try:
            headers = {}
            if self._bearer_token is not None:
                headers["Authorization"] = (
                    f"Bearer {self._bearer_token.get_secret_value()}"
                )
            timeout = httpx2.Timeout(
                connect=self._config.connect_timeout_seconds,
                read=self._config.read_timeout_seconds,
                write=self._config.read_timeout_seconds,
                pool=self._config.connect_timeout_seconds,
            )
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(headers=headers, timeout=timeout)
            )
            transport = streamable_http_client(
                self._config.url,
                http_client=http_client,
            )
            sdk_client = Client(
                transport,
                read_timeout_seconds=self._config.read_timeout_seconds,
            )
            await stack.enter_async_context(sdk_client)
        except (TimeoutError, httpx2.TimeoutException) as exc:
            await stack.aclose()
            raise MCPConnectionError("MCP HTTP connection timed out") from exc
        except SDKMCPError as exc:
            await stack.aclose()
            if is_sdk_timeout(exc):
                raise MCPConnectionError(
                    "MCP HTTP connection timed out"
                ) from exc
            raise MCPConnectionError("MCP HTTP connection failed") from exc
        except TRANSPORT_ERRORS as exc:
            await stack.aclose()
            raise MCPConnectionError("MCP HTTP connection failed") from exc

        self._client = sdk_client
        self._exit_stack = stack
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        stack = self._exit_stack
        self._client = None
        self._exit_stack = None
        if stack is not None:
            await stack.__aexit__(exc_type, exc, traceback)

    async def list_tools(self) -> list[MCPRemoteTool]:
        return await discover_tools(self._connected_client())

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        return await call_tool(self._connected_client(), name, arguments)

    def _connected_client(self) -> Client:
        if self._client is None:
            raise MCPConnectionError("MCP client is not connected")
        return self._client
