from dataclasses import dataclass
import re

from app.mcp.client import MCPClient, MCPToolRegistrationError
from app.mcp.tool_adapter import MCPToolAdapter
from app.tools.registry import ToolRegistry


_SERVER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_REMOTE_NAME_SEPARATOR = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class MCPServerClientBinding:
    server_id: str
    client: MCPClient


@dataclass(frozen=True, slots=True)
class MCPToolComposition:
    tools: tuple[MCPToolAdapter, ...]
    local_names: tuple[str, ...]
    approval_required_actions: frozenset[str]


def _local_tool_name(server_id: str, remote_name: str) -> str:
    normalized = _REMOTE_NAME_SEPARATOR.sub("_", remote_name.strip().lower())
    normalized = normalized.strip("_")
    if not normalized:
        raise MCPToolRegistrationError("Invalid MCP tool name")
    return f"mcp_{server_id}_{normalized}"


async def compose_mcp_tools(
    registry: ToolRegistry,
    bindings: list[MCPServerClientBinding],
) -> MCPToolComposition:
    server_ids: set[str] = set()
    for binding in bindings:
        if not _SERVER_ID_PATTERN.fullmatch(binding.server_id):
            raise MCPToolRegistrationError("Invalid MCP server ID")
        if binding.server_id in server_ids:
            raise MCPToolRegistrationError("Duplicate MCP server ID")
        server_ids.add(binding.server_id)

    discovered = []
    for binding in bindings:
        remote_tools = await binding.client.list_tools()
        discovered.append((binding, remote_tools))

    adapters: list[MCPToolAdapter] = []
    local_names: set[str] = set()
    for binding, remote_tools in discovered:
        for remote_tool in remote_tools:
            local_name = _local_tool_name(binding.server_id, remote_tool.name)
            if local_name in local_names:
                raise MCPToolRegistrationError(
                    "MCP tool names collide after normalization"
                )
            if registry.contains(local_name):
                raise MCPToolRegistrationError(
                    "MCP tool name conflicts with an existing tool"
                )
            local_names.add(local_name)
            adapters.append(
                MCPToolAdapter(
                    binding.client,
                    local_name=local_name,
                    remote_name=remote_tool.name,
                    description=remote_tool.description,
                    input_schema=remote_tool.input_schema,
                )
            )

    for adapter in adapters:
        registry.register(adapter)

    names = tuple(adapter.name for adapter in adapters)
    return MCPToolComposition(
        tools=tuple(adapters),
        local_names=names,
        approval_required_actions=frozenset(names),
    )
