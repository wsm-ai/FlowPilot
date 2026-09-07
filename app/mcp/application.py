from contextlib import AsyncExitStack
import os

from pydantic import SecretStr

from app.mcp.client import MCPClient, MCPConnectionError
from app.mcp.config import (
    MCPConfiguredStdioServerConfig,
    MCPHTTPServerConfig,
    MCPServerConfig,
)
from app.mcp.http_client import StreamableHTTPMCPClient
from app.mcp.stdio_client import MCPStdioServerConfig, StdioMCPClient
from app.mcp.tool_composition import (
    MCPServerClientBinding,
    MCPToolComposition,
    compose_mcp_tools,
)
from app.tools.registry import ToolRegistry


def create_configured_mcp_client(config: MCPServerConfig) -> MCPClient:
    if isinstance(config, MCPConfiguredStdioServerConfig):
        return StdioMCPClient(
            MCPStdioServerConfig(
                command=config.command,
                args=config.args,
                env=config.env,
                cwd=config.cwd,
                read_timeout_seconds=config.read_timeout_seconds,
            )
        )

    token: SecretStr | None = None
    if config.bearer_token_env is not None:
        raw_token = os.environ.get(config.bearer_token_env)
        if raw_token is None or not raw_token.strip():
            raise MCPConnectionError(
                "MCP authentication secret is not configured"
            )
        token = SecretStr(raw_token)
    return StreamableHTTPMCPClient(config, bearer_token=token)


async def compose_configured_mcp_tools(
    registry: ToolRegistry,
    configs: list[MCPServerConfig],
    exit_stack: AsyncExitStack,
) -> MCPToolComposition:
    bindings: list[MCPServerClientBinding] = []
    for config in configs:
        client = create_configured_mcp_client(config)
        connected = await exit_stack.enter_async_context(client)
        bindings.append(MCPServerClientBinding(config.server_id, connected))
    return await compose_mcp_tools(registry, bindings)
