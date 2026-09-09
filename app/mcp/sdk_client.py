from copy import deepcopy
from typing import Any

import anyio
import httpx2
from mcp import MCPError as SDKMCPError
from pydantic import ValidationError

from app.mcp.client import (
    MCPAuthenticationError,
    MCPConnectionConfigurationError,
    MCPDiscoveryValidationError,
    MCPPermanentDiscoveryError,
    MCPToolCallError,
    MCPTransientDiscoveryError,
)
from app.mcp.models import MCPRemoteTool, MCPToolResult


REQUEST_TIMEOUT = -32001
TRANSPORT_ERRORS = (
    OSError,
    anyio.EndOfStream,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    httpx2.TransportError,
    httpx2.HTTPStatusError,
)


def is_sdk_timeout(exc: SDKMCPError) -> bool:
    return exc.code == REQUEST_TIMEOUT


async def discover_tools(client: Any) -> list[MCPRemoteTool]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    seen_names: set[str] = set()
    tools: list[MCPRemoteTool] = []

    while True:
        try:
            response = await client.list_tools(cursor=cursor)
        except (TimeoutError, httpx2.TimeoutException) as exc:
            raise MCPTransientDiscoveryError(
                "MCP tool discovery timed out"
            ) from exc
        except httpx2.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                raise MCPAuthenticationError(
                    "MCP authentication configuration failed"
                ) from exc
            if status in {408, 429} or status >= 500:
                raise MCPTransientDiscoveryError(
                    "MCP tool discovery temporarily unavailable"
                ) from exc
            raise MCPPermanentDiscoveryError(
                "MCP tool discovery failed"
            ) from exc
        except (FileNotFoundError, PermissionError) as exc:
            raise MCPConnectionConfigurationError(
                "MCP connection configuration failed"
            ) from exc
        except SDKMCPError as exc:
            if is_sdk_timeout(exc):
                raise MCPTransientDiscoveryError(
                    "MCP tool discovery timed out"
                ) from exc
            raise MCPPermanentDiscoveryError(
                "MCP tool discovery failed"
            ) from exc
        except TRANSPORT_ERRORS as exc:
            raise MCPTransientDiscoveryError(
                "MCP tool discovery temporarily unavailable"
            ) from exc
        except ValidationError as exc:
            raise MCPDiscoveryValidationError(
                "MCP tool discovery returned invalid data"
            ) from exc

        try:
            for sdk_tool in response.tools:
                remote_tool = MCPRemoteTool(
                    name=sdk_tool.name,
                    description=sdk_tool.description or "",
                    input_schema=deepcopy(sdk_tool.input_schema),
                )
                if remote_tool.name in seen_names:
                    raise MCPDiscoveryValidationError(
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
        except MCPDiscoveryValidationError:
            raise
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            raise MCPDiscoveryValidationError(
                "MCP tool discovery returned invalid data"
            ) from exc


async def call_tool(
    client: Any,
    name: str,
    arguments: dict[str, Any],
) -> MCPToolResult:
    try:
        response = await client.call_tool(name, arguments=deepcopy(arguments))
    except (TimeoutError, httpx2.TimeoutException) as exc:
        raise MCPToolCallError("MCP tool call timed out") from exc
    except SDKMCPError as exc:
        if is_sdk_timeout(exc):
            raise MCPToolCallError("MCP tool call timed out") from exc
        raise MCPToolCallError("MCP tool call failed") from exc
    except TRANSPORT_ERRORS as exc:
        raise MCPToolCallError("MCP tool call failed") from exc
    except ValidationError as exc:
        raise MCPToolCallError("MCP tool call returned invalid data") from exc

    try:
        content = [
            block.model_dump(mode="json", by_alias=True, exclude_none=True)
            for block in response.content
        ]
        return MCPToolResult(
            content=content,
            structured_content=deepcopy(response.structured_content),
            is_error=response.is_error,
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise MCPToolCallError("MCP tool call returned invalid data") from exc
