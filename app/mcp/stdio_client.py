import math
from pathlib import Path
from typing import Any

from mcp import Client, MCPError as SDKMCPError, StdioServerParameters
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.mcp.client import (
    MCPConnectionError,
    MCPConnectionConfigurationError,
    MCPPermanentConnectionError,
    MCPTransientConnectionError,
)
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.sdk_client import (
    TRANSPORT_ERRORS,
    call_tool,
    discover_tools,
    is_sdk_timeout,
)


class MCPStdioServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = Field(default=None, repr=False)
    cwd: str | None = None
    read_timeout_seconds: float | None = 30.0

    @field_validator("command")
    @classmethod
    def command_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("command must not be blank")
        return normalized

    @field_validator("cwd")
    @classmethod
    def normalize_cwd(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("cwd must not be blank")
        return normalized

    @field_validator("read_timeout_seconds")
    @classmethod
    def timeout_must_be_positive_and_finite(
        cls, value: float | None
    ) -> float | None:
        if value is not None and (value <= 0 or not math.isfinite(value)):
            raise ValueError("read_timeout_seconds must be positive and finite")
        return value


class StdioMCPClient:
    def __init__(self, config: MCPStdioServerConfig) -> None:
        self._config = config.model_copy(deep=True)
        self._client: Client | None = None

    async def __aenter__(self) -> "StdioMCPClient":
        if self._client is not None:
            raise MCPConnectionConfigurationError(
                "MCP client is already connected"
            )
        parameters = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=None if self._config.env is None else dict(self._config.env),
            cwd=None if self._config.cwd is None else Path(self._config.cwd),
        )
        sdk_client = Client(
            parameters,
            read_timeout_seconds=self._config.read_timeout_seconds,
        )
        try:
            await sdk_client.__aenter__()
        except TimeoutError as exc:
            raise MCPTransientConnectionError(
                "MCP stdio connection timed out"
            ) from exc
        except (FileNotFoundError, PermissionError) as exc:
            raise MCPConnectionConfigurationError(
                "MCP stdio connection configuration failed"
            ) from exc
        except SDKMCPError as exc:
            if is_sdk_timeout(exc):
                raise MCPTransientConnectionError(
                    "MCP stdio connection timed out"
                ) from exc
            raise MCPPermanentConnectionError(
                "MCP stdio connection failed"
            ) from exc
        except TRANSPORT_ERRORS as exc:
            raise MCPTransientConnectionError(
                "MCP stdio connection failed"
            ) from exc
        self._client = sdk_client
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.__aexit__(exc_type, exc, traceback)

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
            raise MCPConnectionConfigurationError(
                "MCP client is not connected"
            )
        return self._client
