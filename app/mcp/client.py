from typing import Any, Protocol

from app.mcp.models import MCPRemoteTool, MCPToolResult


class MCPError(Exception):
    """Base error for failures at the MCP client boundary."""


class MCPConnectionError(MCPError):
    """Base error for failures while establishing an MCP connection."""


class MCPTransientConnectionError(MCPConnectionError):
    """Raised for explicit temporary MCP availability failures."""


class MCPAuthenticationError(MCPConnectionError):
    """Raised when MCP authentication is missing or rejected."""


class MCPConnectionConfigurationError(MCPConnectionError):
    """Raised when local MCP connection configuration is invalid."""


class MCPPermanentConnectionError(MCPConnectionError):
    """Raised for non-timeout MCP protocol or handshake failures."""


class MCPDiscoveryError(MCPError):
    """Base error for failures while discovering remote MCP tools."""


class MCPTransientDiscoveryError(MCPDiscoveryError):
    """Raised for explicit temporary MCP discovery failures."""


class MCPPermanentDiscoveryError(MCPDiscoveryError):
    """Raised for non-timeout MCP discovery protocol failures."""


class MCPDiscoveryValidationError(MCPDiscoveryError):
    """Raised when MCP discovery returns malformed or invalid data."""


class MCPToolCallError(MCPError):
    """Raised when an MCP client cannot complete a remote tool call."""


class MCPToolRegistrationError(MCPError):
    """Raised when discovered MCP tools cannot be registered safely."""


class MCPClient(Protocol):
    async def list_tools(self) -> list[MCPRemoteTool]:
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        ...
