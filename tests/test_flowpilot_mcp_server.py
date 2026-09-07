import asyncio

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from app.main import app
from app.mcp.models import MCPToolResult
from app.mcp.server import (
    FLOWPILOT_MCP_EXPORT_NAMES,
    FlowPilotMCPServerRuntime,
    MCPServerExportError,
    MCPServerRuntimeError,
    create_flowpilot_mcp_server,
    create_mcp_export_registry,
    flowpilot_mcp_application,
)
from app.mcp.tool_adapter import MCPToolAdapter
from app.retrieval.models import KnowledgeChunk, RetrievalResult
from app.tools.base import ToolExecutionError
from app.tools.customer_feedback import CustomerFeedbackTool
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import ToolRegistry


class FakeRetriever:
    async def search(self, query):
        return [
            RetrievalResult(
                chunk=KnowledgeChunk(
                    chunk_id="CH-001",
                    document_id="DOC-001",
                    content="Reset the enterprise login session.",
                    source="support-handbook",
                    metadata={"department": "support"},
                    position=0,
                ),
                score=0.9,
                rank=1,
            )
        ]


class MarkerTool:
    description = "Marker"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, name, *, error=None):
        self.name = name
        self.error = error

    async def execute(self, arguments):
        if self.error is not None:
            raise self.error
        return {"ok": True}


class FakeExternalClient:
    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):
        return MCPToolResult(structured_content={"proxied": True})


def local_registry():
    registry = ToolRegistry()
    registry.register(CustomerFeedbackTool())
    registry.register(KnowledgeBaseTool(FakeRetriever()))
    return registry


def test_in_memory_client_lists_only_explicit_local_read_only_tools():
    registry = local_registry()
    registry.register(MarkerTool("create_issue"))
    registry.register(MarkerTool("mcp_external_tool"))
    export = create_mcp_export_registry(
        registry,
        approval_required_actions={"create_issue", "mcp_external_tool"},
    )
    runtime = FlowPilotMCPServerRuntime()
    runtime.bind(export)
    server = create_flowpilot_mcp_server(runtime)

    async def scenario():
        async with Client(server) as client:
            return await client.list_tools()

    response = asyncio.run(scenario())
    names = {tool.name for tool in response.tools}
    assert names == FLOWPILOT_MCP_EXPORT_NAMES
    assert names.isdisjoint(
        {
            "mcp_external_tool",
            "create_issue",
            "run_agent",
            "planned_agent",
            "resume_approval",
            "answer_retry",
        }
    )


def test_in_memory_client_calls_existing_registry_business_tools():
    runtime = FlowPilotMCPServerRuntime()
    runtime.bind(create_mcp_export_registry(local_registry(), frozenset()))
    server = create_flowpilot_mcp_server(runtime)

    async def scenario():
        async with Client(server) as client:
            feedback = await client.call_tool(
                "get_customer_feedback", {"customer_id": "C001"}
            )
            knowledge = await client.call_tool(
                "search_knowledge_base",
                {"query": "login", "top_k": 1},
            )
            return feedback, knowledge

    feedback, knowledge = asyncio.run(scenario())
    assert feedback.is_error is False
    assert feedback.structured_content is not None
    assert "FB-001" in str(feedback.structured_content)
    assert knowledge.is_error is False
    assert knowledge.structured_content is not None
    assert "CH-001" in str(knowledge.structured_content)


def test_runtime_bind_unbind_and_unbound_call_fail_closed():
    runtime = FlowPilotMCPServerRuntime()
    export = create_mcp_export_registry(local_registry(), frozenset())
    runtime.bind(export)
    assert runtime.is_bound is True
    with pytest.raises(MCPServerRuntimeError, match="already initialized"):
        runtime.bind(export)
    runtime.unbind()
    assert runtime.is_bound is False
    server = create_flowpilot_mcp_server(runtime)

    async def scenario():
        async with Client(server) as client:
            return await client.call_tool(
                "get_customer_feedback", {"customer_id": "C001"}
            )

    result = asyncio.run(scenario())
    assert result.is_error is True
    assert "FlowPilot MCP server is not initialized" in str(result.content)


def test_tool_execution_error_is_sanitized_at_server_boundary():
    registry = ToolRegistry()
    registry.register(
        MarkerTool(
            "get_customer_feedback",
            error=ToolExecutionError("DATABASE_PASSWORD=super-secret"),
        )
    )
    registry.register(MarkerTool("search_knowledge_base"))
    runtime = FlowPilotMCPServerRuntime()
    runtime.bind(create_mcp_export_registry(registry, frozenset()))
    server = create_flowpilot_mcp_server(runtime)

    async def scenario():
        async with Client(server) as client:
            return await client.call_tool(
                "get_customer_feedback", {"customer_id": "C001"}
            )

    result = asyncio.run(scenario())
    serialized = str(result.content)
    assert result.is_error is True
    assert "FlowPilot tool execution failed" in serialized
    assert "super-secret" not in serialized
    assert "DATABASE_PASSWORD" not in serialized


def test_export_policy_rejects_missing_and_approval_required_tools():
    registry = local_registry()
    with pytest.raises(MCPServerExportError, match="unavailable"):
        create_mcp_export_registry(
            registry, frozenset(), {"missing_tool"}
        )
    with pytest.raises(MCPServerExportError, match="Approval-required"):
        create_mcp_export_registry(
            registry,
            {"search_knowledge_base"},
            {"search_knowledge_base"},
        )


def test_mcp_tool_adapter_cannot_be_reexported_and_full_registry_is_unchanged():
    registry = local_registry()
    adapter = MCPToolAdapter(
        FakeExternalClient(),
        local_name="mcp_external_tool",
        remote_name="remote_tool",
        description="External tool",
        input_schema={"type": "object"},
    )
    registry.register(adapter)

    with pytest.raises(MCPServerExportError, match="External MCP tools"):
        create_mcp_export_registry(
            registry, frozenset(), {"mcp_external_tool"}
        )

    assert registry.contains("mcp_external_tool")
    safe_export = create_mcp_export_registry(registry, frozenset())
    assert not safe_export.contains("mcp_external_tool")


def test_fastapi_lifespan_binds_runtime_runs_session_manager_and_unbinds(
):
    assert flowpilot_mcp_application.runtime.is_bound is False
    with TestClient(app) as client:
        assert flowpilot_mcp_application.runtime.is_bound is True
        assert flowpilot_mcp_application.session_manager_running is True
        assert client.get("/health").status_code == 200
    assert flowpilot_mcp_application.runtime.is_bound is False
    assert flowpilot_mcp_application.session_manager_running is False


def test_mcp_mount_uses_single_path_and_no_legacy_sse():
    mount = next(
        route for route in app.routes if getattr(route, "path", None) == "/mcp"
    )
    child_paths = {route.path for route in mount.app.asgi_app.routes}
    assert "/" in child_paths
    assert "/mcp" not in child_paths
    assert not any(getattr(route, "path", "") == "/sse" for route in app.routes)
