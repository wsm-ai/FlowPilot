from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.mcp.tool_adapter import MCPToolAdapter
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


FLOWPILOT_MCP_EXPORT_NAMES = frozenset(
    {"get_customer_feedback", "search_knowledge_base"}
)


class MCPServerExportError(Exception):
    """Raised when the MCP server export policy cannot be applied safely."""


class MCPServerRuntimeError(Exception):
    """Raised when the FlowPilot MCP server runtime lifecycle is invalid."""


class FlowPilotMCPServerRuntime:
    def __init__(self) -> None:
        self._registry: ToolRegistry | None = None

    @property
    def is_bound(self) -> bool:
        return self._registry is not None

    def bind(self, registry: ToolRegistry) -> None:
        if self._registry is not None:
            raise MCPServerRuntimeError("FlowPilot MCP server is already initialized")
        self._registry = registry

    def unbind(self) -> None:
        self._registry = None

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        registry = self._registry
        if registry is None:
            raise ToolError("FlowPilot MCP server is not initialized")
        try:
            return await registry.execute(name, arguments)
        except ToolExecutionError as exc:
            raise ToolError("FlowPilot tool execution failed") from exc


def create_mcp_export_registry(
    full_registry: ToolRegistry,
    approval_required_actions: set[str] | frozenset[str],
    export_names: set[str] | frozenset[str] = FLOWPILOT_MCP_EXPORT_NAMES,
) -> ToolRegistry:
    requested = frozenset(export_names)
    if requested & frozenset(approval_required_actions):
        raise MCPServerExportError(
            "Approval-required tools cannot be exported by the MCP server"
        )
    try:
        export_registry = full_registry.including(requested)
    except ToolExecutionError as exc:
        raise MCPServerExportError(
            "Configured MCP server export tool is unavailable"
        ) from exc
    for name in requested:
        if isinstance(export_registry.get(name), MCPToolAdapter):
            raise MCPServerExportError(
                "External MCP tools cannot be exported by the MCP server"
            )
    return export_registry


def create_flowpilot_mcp_server(
    runtime: FlowPilotMCPServerRuntime,
) -> MCPServer:
    server = MCPServer(
        "FlowPilot",
        description="FlowPilot local read-only enterprise tools.",
    )

    @server.tool(
        name="get_customer_feedback",
        description=(
            "Get customer feedback records by customer ID and optional priority."
        ),
    )
    async def get_customer_feedback(
        customer_id: str,
        priority: Literal["low", "medium", "high"] | None = None,
    ) -> list[dict[str, str]]:
        return await runtime.execute(
            "get_customer_feedback",
            {"customer_id": customer_id, "priority": priority},
        )

    @server.tool(
        name="search_knowledge_base",
        description=(
            "Search the internal knowledge base for evidence relevant to a question "
            "or workflow step. Returns ranked knowledge chunks with source metadata."
        ),
    )
    async def search_knowledge_base(
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return await runtime.execute(
            "search_knowledge_base",
            {
                "query": query,
                "top_k": top_k,
                "filters": {} if filters is None else filters,
            },
        )

    return server


class FlowPilotMCPServerApplication:
    def __init__(self) -> None:
        self.runtime = FlowPilotMCPServerRuntime()
        self.server: MCPServer
        self.asgi_app: Any
        self._session_manager_running = False
        self.reset_server()

    @property
    def session_manager_running(self) -> bool:
        return self._session_manager_running

    def reset_server(self) -> None:
        if self._session_manager_running:
            raise MCPServerRuntimeError(
                "FlowPilot MCP server session manager is already running"
            )
        self.server = create_flowpilot_mcp_server(self.runtime)
        self.asgi_app = self.server.streamable_http_app(
            streamable_http_path="/"
        )

    @asynccontextmanager
    async def run_session_manager(self):
        if self._session_manager_running:
            raise MCPServerRuntimeError(
                "FlowPilot MCP server session manager is already running"
            )
        self._session_manager_running = True
        try:
            async with self.server.session_manager.run():
                yield
        finally:
            self._session_manager_running = False

    async def __call__(self, scope, receive, send) -> None:
        await self.asgi_app(scope, receive, send)


def build_flowpilot_mcp_server_application() -> FlowPilotMCPServerApplication:
    return FlowPilotMCPServerApplication()


flowpilot_mcp_application = build_flowpilot_mcp_server_application()
