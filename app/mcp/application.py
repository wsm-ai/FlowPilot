from contextlib import AsyncExitStack
import os

from pydantic import SecretStr

from app.mcp.client import MCPAuthenticationError, MCPClient
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
from app.reliability.degradation import (
    DegradationRecord,
    decide_degradation,
)
from app.reliability.failures import classify_failure
from app.reliability.retry import OperationSemantics
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
            raise MCPAuthenticationError(
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
    degradations: list[DegradationRecord] = []
    for config in configs:
        client = create_configured_mcp_client(config)
        try:
            connected = await exit_stack.enter_async_context(client)
        except Exception as exc:
            failure = classify_failure(exc)
            decision = decide_degradation(
                failure,
                OperationSemantics.READ_ONLY,
                optional=not config.required,
            )
            if not decision.allowed:
                raise
            degradations.append(
                DegradationRecord.from_failure(config.server_id, failure)
            )
            continue
        bindings.append(
            MCPServerClientBinding(
                config.server_id,
                connected,
                required=config.required,
            )
        )
    composition = await compose_mcp_tools(registry, bindings)
    return MCPToolComposition(
        tools=composition.tools,
        local_names=composition.local_names,
        approval_required_actions=composition.approval_required_actions,
        degradations=tuple(degradations) + composition.degradations,
    )
