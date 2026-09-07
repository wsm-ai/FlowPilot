from typing import Any, Protocol

from app.mcp.models import MCPRemoteTool, MCPToolResult


class MCPError(Exception):
    """Base error for failures at the MCP client boundary."""


class MCPConnectionError(MCPError):
    """Raised when an MCP client cannot reach its server."""


class MCPDiscoveryError(MCPError):
    """Raised when an MCP client cannot discover remote tools."""


class MCPToolCallError(MCPError):
    """Raised when an MCP client cannot complete a remote tool call."""


class MCPClient(Protocol):
    async def list_tools(self) -> list[MCPRemoteTool]:
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        ...
