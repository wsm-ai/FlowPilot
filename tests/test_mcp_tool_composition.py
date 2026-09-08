import asyncio
import json
from pathlib import Path
import sys

import pytest

from app.mcp.client import MCPDiscoveryError, MCPToolRegistrationError
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.stdio_client import MCPStdioServerConfig, StdioMCPClient
from app.mcp.tool_adapter import MCPToolAdapter
from app.mcp.tool_composition import (
    MCPServerClientBinding,
    compose_mcp_tools,
)
from app.persistence.checkpoint import async_checkpoint_saver
from app.providers.types import LLMResponse
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.registry import ToolRegistry, create_default_tool_registry


SERVER_PATH = (
    Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
).resolve()


def remote_tool(name: str) -> MCPRemoteTool:
    return MCPRemoteTool(
        name=name,
        description=f"Remote {name}",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


class FakeMCPClient:
    def __init__(self, tools, *, discovery_error=None, result=None):
        self.tools = tools
        self.discovery_error = discovery_error
        self.result = result or MCPToolResult(structured_content={"ok": True})
        self.call_tool_calls = []
        self.list_tools_call_count = 0

    async def list_tools(self):
        self.list_tools_call_count += 1
        if self.discovery_error is not None:
            raise self.discovery_error
        return self.tools

    async def call_tool(self, name, arguments):
        self.call_tool_calls.append((name, arguments))
        return self.result


class FakeProvider:
    model = "test-model"

    def __init__(self, plan):
        self.plan = plan

    async def complete(self, messages, tools=None, tool_choice=None):
        return LLMResponse(content=json.dumps(self.plan))


def test_namespaces_normalizes_and_preserves_remote_call_name():
    client = FakeMCPClient([remote_tool("Search-Issues")])
    registry = create_default_tool_registry()

    composition = asyncio.run(
        compose_mcp_tools(
            registry,
            [MCPServerClientBinding("github", client)],
        )
    )
    result = asyncio.run(
        registry.execute("mcp_github_search_issues", {"query": "login"})
    )

    assert composition.local_names == ("mcp_github_search_issues",)
    assert composition.approval_required_actions == {
        "mcp_github_search_issues"
    }
    assert client.call_tool_calls == [("Search-Issues", {"query": "login"})]
    assert result == {"ok": True}
    names = [item["function"]["name"] for item in registry.definitions()]
    assert names == ["get_customer_feedback", "mcp_github_search_issues"]
    assert registry.definitions()[1]["function"]["parameters"] == (
        remote_tool("Search-Issues").input_schema
    )


def test_same_remote_name_from_different_servers_is_isolated():
    github = FakeMCPClient([remote_tool("search")])
    linear = FakeMCPClient([remote_tool("search")])
    registry = ToolRegistry()

    composition = asyncio.run(
        compose_mcp_tools(
            registry,
            [
                MCPServerClientBinding("github", github),
                MCPServerClientBinding("linear", linear),
            ],
        )
    )
    asyncio.run(registry.execute("mcp_github_search", {"query": "a"}))
    asyncio.run(registry.execute("mcp_linear_search", {"query": "b"}))

    assert composition.local_names == (
        "mcp_github_search",
        "mcp_linear_search",
    )
    assert github.call_tool_calls == [("search", {"query": "a"})]
    assert linear.call_tool_calls == [("search", {"query": "b"})]


@pytest.mark.parametrize("server_id", ["", "GitHub Prod", "../github", "github-prod"])
def test_invalid_server_id_is_rejected_without_discovery(server_id):
    client = FakeMCPClient([remote_tool("search")])
    with pytest.raises(MCPToolRegistrationError, match="Invalid MCP server ID"):
        asyncio.run(
            compose_mcp_tools(
                ToolRegistry(),
                [MCPServerClientBinding(server_id, client)],
            )
        )


def test_normalization_collision_is_atomic():
    registry = ToolRegistry()
    client = FakeMCPClient(
        [remote_tool("search-issues"), remote_tool("search_issues")]
    )

    with pytest.raises(
        MCPToolRegistrationError,
        match="MCP tool names collide after normalization",
    ):
        asyncio.run(
            compose_mcp_tools(
                registry,
                [MCPServerClientBinding("github", client)],
            )
        )

    assert registry.definitions() == []


def test_existing_registry_collision_is_atomic():
    registry = ToolRegistry()
    existing_client = FakeMCPClient([])
    registry.register(
        MCPToolAdapter(
            existing_client,
            local_name="mcp_github_search",
            remote_name="existing",
            description="Existing",
            input_schema={"type": "object"},
        )
    )
    client = FakeMCPClient([remote_tool("other"), remote_tool("search")])

    with pytest.raises(MCPToolRegistrationError, match="existing tool"):
        asyncio.run(
            compose_mcp_tools(
                registry,
                [MCPServerClientBinding("github", client)],
            )
        )

    assert [item["function"]["name"] for item in registry.definitions()] == [
        "mcp_github_search"
    ]


def test_duplicate_server_id_is_rejected_atomically():
    registry = ToolRegistry()
    with pytest.raises(MCPToolRegistrationError, match="Duplicate MCP server ID"):
        asyncio.run(
            compose_mcp_tools(
                registry,
                [
                    MCPServerClientBinding(
                        "github", FakeMCPClient([remote_tool("search")])
                    ),
                    MCPServerClientBinding(
                        "github", FakeMCPClient([remote_tool("create")])
                    ),
                ],
            )
        )
    assert registry.definitions() == []


def test_discovery_failure_does_not_partially_register():
    registry = ToolRegistry()
    failing_client = FakeMCPClient(
        [], discovery_error=MCPDiscoveryError("unavailable")
    )
    with pytest.raises(MCPDiscoveryError, match="unavailable"):
        asyncio.run(
            compose_mcp_tools(
                registry,
                [
                    MCPServerClientBinding(
                        "github", FakeMCPClient([remote_tool("search")])
                    ),
                    MCPServerClientBinding(
                        "linear",
                        failing_client,
                    ),
                ],
            )
        )
    assert registry.definitions() == []
    assert failing_client.list_tools_call_count == 2


def test_discovery_transient_failure_retries_once_then_registers():
    class RecoveringClient(FakeMCPClient):
        async def list_tools(self):
            self.list_tools_call_count += 1
            if self.list_tools_call_count == 1:
                raise MCPDiscoveryError("temporarily unavailable")
            return self.tools

    registry = ToolRegistry()
    client = RecoveringClient([remote_tool("search")])

    composition = asyncio.run(
        compose_mcp_tools(
            registry,
            [MCPServerClientBinding("github", client)],
        )
    )

    assert client.list_tools_call_count == 2
    assert composition.local_names == ("mcp_github_search",)
    assert registry.contains("mcp_github_search")


def test_unexpected_discovery_error_is_not_wrapped():
    error = RuntimeError("unexpected bug")
    with pytest.raises(RuntimeError, match="unexpected bug") as raised:
        asyncio.run(
            compose_mcp_tools(
                ToolRegistry(),
                [
                    MCPServerClientBinding(
                        "github", FakeMCPClient([], discovery_error=error)
                    )
                ],
            )
        )
    assert raised.value is error


def test_planner_forces_composed_mcp_action_to_require_approval():
    client = FakeMCPClient([remote_tool("search_issues")])
    registry = ToolRegistry()
    composition = asyncio.run(
        compose_mcp_tools(
            registry,
            [MCPServerClientBinding("github", client)],
        )
    )
    plan_data = {
        "goal": "Search issues",
        "steps": [
            {
                "id": 1,
                "description": "Search remote issues",
                "action": "mcp_github_search_issues",
                "arguments": {"query": "login"},
                "requires_approval": False,
            }
        ],
    }
    planner = PlannerService(
        LLMService(FakeProvider(plan_data)),
        registry.definitions(),
        composition.approval_required_actions,
    )

    plan = asyncio.run(planner.create_plan("Search issues"))

    assert plan.steps[0].action == "mcp_github_search_issues"
    assert plan.steps[0].requires_approval is True


def test_unknown_namespaced_action_is_still_rejected():
    plan_data = {
        "goal": "Delete everything",
        "steps": [
            {
                "id": 1,
                "description": "Unknown remote action",
                "action": "mcp_unknown_delete_everything",
            }
        ],
    }
    planner = PlannerService(
        LLMService(FakeProvider(plan_data)),
        [],
        {"mcp_unknown_delete_everything"},
    )
    with pytest.raises(PlanningError, match="Plan contains unavailable action"):
        asyncio.run(planner.create_plan("Delete everything"))


def test_conflicting_approval_policy_fails_fast():
    with pytest.raises(PlanningError, match="Conflicting tool approval policy"):
        PlannerService(
            LLMService(FakeProvider({})),
            [],
            {"search_knowledge_base"},
        )


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_calls"),
    [("approve", "completed", 1), ("reject", "rejected", 0)],
)
def test_mcp_action_obeys_existing_hitl_lifecycle(
    tmp_path, decision, expected_status, expected_calls
):
    async def scenario():
        client = FakeMCPClient([remote_tool("create_issue")])
        registry = ToolRegistry()
        composition = await compose_mcp_tools(
            registry,
            [MCPServerClientBinding("github", client)],
        )
        plan_data = {
            "goal": "Create issue",
            "steps": [
                {
                    "id": 1,
                    "description": "Create remote issue",
                    "action": "mcp_github_create_issue",
                    "arguments": {"query": "Login bug"},
                    "requires_approval": False,
                }
            ],
        }
        planner = PlannerService(
            LLMService(FakeProvider(plan_data)),
            registry.definitions(),
            composition.approval_required_actions,
        )
        async with async_checkpoint_saver(tmp_path / f"{decision}.sqlite") as saver:
            service = ApprovalWorkflowService(planner, registry, saver)
            started = await service.start("Create issue", thread_id="mcp-thread")
            calls_before_resume = list(client.call_tool_calls)
            finished = await service.resume("mcp-thread", decision)
        return started, calls_before_resume, finished, client.call_tool_calls

    started, before, finished, calls = asyncio.run(scenario())

    assert started.status == "approval_required"
    assert started.plan.steps[0].requires_approval is True
    assert before == []
    assert finished.status == expected_status
    assert len(calls) == expected_calls
    if decision == "approve":
        assert calls == [("create_issue", {"query": "Login bug"})]


def test_real_stdio_discovery_composition_and_registry_execution():
    async def scenario():
        config = MCPStdioServerConfig(
            command=sys.executable,
            args=[str(SERVER_PATH)],
        )
        async with StdioMCPClient(config) as client:
            registry = ToolRegistry()
            composition = await compose_mcp_tools(
                registry,
                [MCPServerClientBinding("test", client)],
            )
            result = await registry.execute(
                "mcp_test_add_numbers", {"a": 2, "b": 5}
            )
            return composition, result

    composition, result = asyncio.run(scenario())

    assert "mcp_test_add_numbers" in composition.local_names
    assert result == {"sum": 7}
