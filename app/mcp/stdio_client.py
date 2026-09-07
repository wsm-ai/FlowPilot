from copy import deepcopy
from pathlib import Path
from typing import Any

from mcp import Client, MCPError as SDKMCPError, StdioServerParameters
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.mcp.client import (
    MCPConnectionError,
    MCPDiscoveryError,
    MCPToolCallError,
)
from app.mcp.models import MCPRemoteTool, MCPToolResult


class MCPStdioServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = Field(default=None, repr=False)
    cwd: str | None = None

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


class StdioMCPClient:
    def __init__(self, config: MCPStdioServerConfig) -> None:
        self._config = config.model_copy(deep=True)
        self._client: Client | None = None

    async def __aenter__(self) -> "StdioMCPClient":
        if self._client is not None:
            raise MCPConnectionError("MCP client is already connected")
        parameters = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=None if self._config.env is None else dict(self._config.env),
            cwd=None if self._config.cwd is None else Path(self._config.cwd),
        )
        sdk_client = Client(parameters)
        try:
            await sdk_client.__aenter__()
        except (OSError, SDKMCPError) as exc:
            raise MCPConnectionError("MCP stdio connection failed") from exc
        self._client = sdk_client
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.__aexit__(exc_type, exc, traceback)

    async def list_tools(self) -> list[MCPRemoteTool]:
        client = self._connected_client()
        cursor: str | None = None
        seen_cursors: set[str] = set()
        seen_names: set[str] = set()
        tools: list[MCPRemoteTool] = []

        while True:
            try:
                response = await client.list_tools(cursor=cursor)
            except (OSError, SDKMCPError) as exc:
                raise MCPDiscoveryError("MCP tool discovery failed") from exc

            try:
                for sdk_tool in response.tools:
                    remote_tool = MCPRemoteTool(
                        name=sdk_tool.name,
                        description=sdk_tool.description or "",
                        input_schema=deepcopy(sdk_tool.input_schema),
                    )
                    if remote_tool.name in seen_names:
                        raise MCPDiscoveryError(
                            "MCP tool discovery returned duplicate tool name"
                        )
                    seen_names.add(remote_tool.name)
                    tools.append(remote_tool)
                next_cursor = response.next_cursor
                if next_cursor is None:
                    return tools
                if next_cursor in seen_cursors:
                    raise ValueError("repeated pagination cursor")
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            except MCPDiscoveryError:
                raise
            except (AttributeError, TypeError, ValueError, ValidationError) as exc:
                raise MCPDiscoveryError(
                    "MCP tool discovery returned invalid data"
                ) from exc

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        client = self._connected_client()
        try:
            response = await client.call_tool(
                name,
                arguments=deepcopy(arguments),
            )
        except (OSError, SDKMCPError) as exc:
            raise MCPToolCallError("MCP tool call failed") from exc

        try:
            content = [
                block.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                )
                for block in response.content
            ]
            structured_content = deepcopy(response.structured_content)
            return MCPToolResult(
                content=content,
                structured_content=structured_content,
                is_error=response.is_error,
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise MCPToolCallError("MCP tool call failed") from exc

    def _connected_client(self) -> Client:
        if self._client is None:
            raise MCPConnectionError("MCP client is not connected")
        return self._client
