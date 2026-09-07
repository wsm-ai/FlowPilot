import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from app.api.agent import (
    get_approval_workflow_service,
    get_graph_agent_service,
    get_planned_agent_service,
)
from app.api.dependencies import (
    get_reactive_tool_registry,
    get_tool_registry,
)
from app.mcp.models import MCPToolResult
from app.mcp.tool_adapter import MCPToolAdapter
from app.providers.types import LLMResponse, LLMToolCall
from app.services.llm_service import LLMService
from app.tools.base import ToolExecutionError
from app.tools.registry import create_default_tool_registry


APPROVAL_TOOL = "mcp_github_create_issue"


class FakeMCPClient:
    def __init__(self):
        self.call_tool_calls = []

    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):
        self.call_tool_calls.append((name, arguments))
        return MCPToolResult(structured_content={"created": True})


class ReactiveProvider:
    model = "test-model"

    def __init__(self):
        self.requests = []

    async def complete(self, messages, tools=None, tool_choice=None):
        self.requests.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        return LLMResponse(
            content=None,
            tool_calls=[
                LLMToolCall(
                    id="call-1",
                    name=APPROVAL_TOOL,
                    arguments='{"title":"Unsafe issue"}',
                )
            ]
        )


class PlanningProvider:
    model = "test-model"

    async def complete(self, messages, tools=None, tool_choice=None):
        return LLMResponse(
            content=json.dumps(
                {
                    "goal": "Create issue",
                    "steps": [
                        {
                            "id": 1,
                            "description": "Create issue",
                            "action": APPROVAL_TOOL,
                            "arguments": {"title": "Issue"},
                            "requires_approval": False,
                        }
                    ],
                }
            )
        )


def full_registry():
    client = FakeMCPClient()
    registry = create_default_tool_registry()
    registry.register(
        MCPToolAdapter(
            client,
            local_name=APPROVAL_TOOL,
            remote_name="create_issue",
            description="Create a remote issue",
            input_schema={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        )
    )
    return registry, client


def request_with_registry(registry):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                tool_registry=registry,
                mcp_approval_required_actions=frozenset({APPROVAL_TOOL}),
            )
        )
    )


def test_reactive_agent_neither_sees_nor_executes_approval_required_tool():
    registry, client = full_registry()
    restricted = get_reactive_tool_registry(request_with_registry(registry))
    provider = ReactiveProvider()
    service = get_graph_agent_service(
        llm_service=LLMService(provider),
        checkpointer=None,
        registry=restricted,
    )

    with pytest.raises(ToolExecutionError, match="Unknown tool"):
        asyncio.run(service.run("Create an issue"))

    exposed_names = {
        definition["function"]["name"]
        for definition in provider.requests[0]["tools"]
    }
    assert APPROVAL_TOOL not in exposed_names
    assert client.call_tool_calls == []
    assert registry.contains(APPROVAL_TOOL)
    assert not restricted.contains(APPROVAL_TOOL)


def test_production_service_dependencies_use_correct_registry_boundaries():
    graph_dependency = inspect.signature(get_graph_agent_service).parameters[
        "registry"
    ].default.dependency
    planned_dependency = inspect.signature(get_planned_agent_service).parameters[
        "registry"
    ].default.dependency
    approval_dependency = inspect.signature(
        get_approval_workflow_service
    ).parameters["registry"].default.dependency

    assert graph_dependency is get_reactive_tool_registry
    assert planned_dependency is get_tool_registry
    assert approval_dependency is get_tool_registry


def test_planned_service_keeps_full_registry_and_forces_approval():
    registry, _ = full_registry()
    service = get_planned_agent_service(
        llm_service=LLMService(PlanningProvider()),
        registry=registry,
        approval_required_actions=frozenset({APPROVAL_TOOL}),
    )

    plan = asyncio.run(service._planner_service.create_plan("Create issue"))

    assert registry.contains(APPROVAL_TOOL)
    assert plan.steps[0].action == APPROVAL_TOOL
    assert plan.steps[0].requires_approval is True
