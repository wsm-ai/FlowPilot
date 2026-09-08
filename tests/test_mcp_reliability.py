import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from mcp import MCPError as SDKMCPError
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ValidationError

import app.mcp.stdio_client as stdio_module
from app.mcp.client import (
    MCPConnectionError,
    MCPDiscoveryError,
    MCPToolCallError,
)
from app.mcp.models import MCPToolResult
from app.mcp.stdio_client import MCPStdioServerConfig, StdioMCPClient
from app.mcp.tool_composition import MCPServerClientBinding, compose_mcp_tools
from app.persistence.checkpoint import async_checkpoint_saver
from app.providers.types import LLMResponse
from app.services.approval_workflow_service import ApprovalWorkflowService
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


SERVER_PATH = (
    Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
).resolve()
REQUEST_TIMEOUT = -32001


def server_config(timeout=30.0):
    return MCPStdioServerConfig(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        read_timeout_seconds=timeout,
    )


def sdk_tool(name):
    return Tool(
        name=name,
        description=f"Description for {name}",
        input_schema={"type": "object", "properties": {}},
    )


class FakeSDKClient:
    enter_error = None
    pages = {
        None: SimpleNamespace(tools=[sdk_tool("remote_action")], next_cursor=None)
    }
    tool_result = CallToolResult(
        content=[TextContent(text="ok")],
        structured_content={"ok": True},
        is_error=False,
    )
    list_error_by_cursor = {}
    call_error = None
    instances = []

    def __init__(self, parameters, *, read_timeout_seconds=None):
        self.parameters = parameters
        self.read_timeout_seconds = read_timeout_seconds
        self.list_calls = []
        self.call_tool_calls = []
        type(self).instances.append(self)

    async def __aenter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def list_tools(self, *, cursor=None):
        self.list_calls.append(cursor)
        error = self.list_error_by_cursor.get(cursor)
        if error is not None:
            raise error
        return self.pages[cursor]

    async def call_tool(self, name, arguments=None):
        self.call_tool_calls.append((name, arguments))
        if self.call_error is not None:
            raise self.call_error
        return self.tool_result


@pytest.fixture(autouse=True)
def reset_fake_sdk_state():
    FakeSDKClient.enter_error = None
    FakeSDKClient.pages = {
        None: SimpleNamespace(tools=[sdk_tool("remote_action")], next_cursor=None)
    }
    FakeSDKClient.tool_result = CallToolResult(
        content=[TextContent(text="ok")],
        structured_content={"ok": True},
        is_error=False,
    )
    FakeSDKClient.list_error_by_cursor = {}
    FakeSDKClient.call_error = None
    FakeSDKClient.instances = []


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf])
def test_timeout_config_rejects_non_positive_or_non_finite_values(value):
    with pytest.raises(ValidationError):
        server_config(value)


def test_timeout_config_accepts_none_and_defaults_to_thirty_seconds():
    assert server_config().read_timeout_seconds == 30.0
    assert server_config(None).read_timeout_seconds is None


def test_timeout_is_passed_to_current_sdk_client(monkeypatch):
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config(4.25)):
            return FakeSDKClient.instances[0].read_timeout_seconds

    assert asyncio.run(scenario()) == 4.25


def test_connection_timeout_maps_to_safe_error(monkeypatch):
    FakeSDKClient.enter_error = SDKMCPError(
        REQUEST_TIMEOUT, "secret connection timeout detail"
    )
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        with pytest.raises(MCPConnectionError) as raised:
            async with StdioMCPClient(server_config()):
                pass
        return raised.value

    error = asyncio.run(scenario())
    assert str(error) == "MCP stdio connection timed out"
    assert "secret" not in str(error)


def test_list_tools_timeout_is_safe_and_not_retried(monkeypatch):
    FakeSDKClient.list_error_by_cursor = {
        None: SDKMCPError(REQUEST_TIMEOUT, "sensitive discovery timeout")
    }
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            with pytest.raises(MCPDiscoveryError) as raised:
                await client.list_tools()
            return raised.value, client._client.list_calls

    error, calls = asyncio.run(scenario())
    assert str(error) == "MCP tool discovery timed out"
    assert "sensitive" not in str(error)
    assert calls == [None]


def test_call_tool_timeout_is_safe_and_not_retried(monkeypatch):
    FakeSDKClient.call_error = SDKMCPError(
        REQUEST_TIMEOUT, "sensitive tool timeout"
    )
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            with pytest.raises(MCPToolCallError) as raised:
                await client.call_tool("remote_action", {})
            return raised.value, client._client.call_tool_calls

    error, calls = asyncio.run(scenario())
    assert str(error) == "MCP tool call timed out"
    assert "sensitive" not in str(error)
    assert calls == [("remote_action", {})]


def test_error_result_is_returned_instead_of_raised(monkeypatch):
    FakeSDKClient.tool_result = CallToolResult(
        content=[TextContent(text="remote failure")],
        is_error=True,
    )
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            return await client.call_tool("remote_action", {})

    result = asyncio.run(scenario())
    assert isinstance(result, MCPToolResult)
    assert result.is_error is True


def test_malformed_call_result_has_distinct_safe_error(monkeypatch):
    FakeSDKClient.tool_result = SimpleNamespace(
        content=[object()], structured_content=None, is_error=False
    )
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            with pytest.raises(MCPToolCallError) as raised:
                await client.call_tool("remote_action", {})
            return raised.value

    assert str(asyncio.run(scenario())) == (
        "MCP tool call returned invalid data"
    )


@pytest.mark.parametrize(
    "error",
    [asyncio.CancelledError(), RuntimeError("unexpected bug")],
)
def test_non_boundary_errors_propagate(monkeypatch, error):
    FakeSDKClient.call_error = error
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            await client.call_tool("remote_action", {})

    with pytest.raises(type(error), match=None if isinstance(error, asyncio.CancelledError) else "unexpected bug"):
        asyncio.run(scenario())


def test_second_discovery_page_timeout_does_not_partially_register(monkeypatch):
    FakeSDKClient.pages = {
        None: SimpleNamespace(tools=[sdk_tool("page_one")], next_cursor="page-2")
    }
    FakeSDKClient.list_error_by_cursor = {
        "page-2": SDKMCPError(REQUEST_TIMEOUT, "private page detail")
    }
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        registry = ToolRegistry()
        async with StdioMCPClient(server_config()) as client:
            with pytest.raises(
                MCPDiscoveryError, match="MCP tool discovery timed out"
            ):
                await compose_mcp_tools(
                    registry,
                    [MCPServerClientBinding("test", client)],
                )
        return registry.definitions(), FakeSDKClient.instances[0].list_calls

    definitions, calls = asyncio.run(scenario())
    assert definitions == []
    assert calls == [None, "page-2", None, "page-2"]


def test_real_slow_tool_times_out_without_retry():
    async def scenario():
        async with StdioMCPClient(server_config(0.05)) as client:
            with pytest.raises(MCPToolCallError) as raised:
                await client.call_tool("slow_tool", {"delay_seconds": 1.0})
            return raised.value

    assert str(asyncio.run(scenario())) == "MCP tool call timed out"


def test_real_hard_exit_maps_crash_and_does_not_hang():
    async def scenario():
        async with asyncio.timeout(3):
            async with StdioMCPClient(server_config(0.5)) as client:
                with pytest.raises(MCPToolCallError) as raised:
                    await client.call_tool("hard_exit", {})
                return raised.value

    assert str(asyncio.run(scenario())) == "MCP tool call failed"


class PlanProvider:
    model = "test-model"

    async def complete(self, messages, tools=None, tool_choice=None):
        return LLMResponse(
            content=json.dumps(
                {
                    "goal": "Run remote action",
                    "steps": [
                        {
                            "id": 1,
                            "description": "Run remote action",
                            "action": "mcp_test_remote_action",
                            "arguments": {},
                            "requires_approval": False,
                        }
                    ],
                }
            )
        )


def test_hitl_timeout_calls_remote_side_effect_only_after_approval_once(
    monkeypatch, tmp_path
):
    FakeSDKClient.call_error = SDKMCPError(
        REQUEST_TIMEOUT, "sensitive side effect timeout"
    )
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            registry = ToolRegistry()
            composition = await compose_mcp_tools(
                registry,
                [MCPServerClientBinding("test", client)],
            )
            planner = PlannerService(
                LLMService(PlanProvider()),
                registry.definitions(),
                composition.approval_required_actions,
            )
            async with async_checkpoint_saver(tmp_path / "hitl.sqlite") as saver:
                service = ApprovalWorkflowService(planner, registry, saver)
                started = await service.start(
                    "Run remote action", thread_id="timeout-thread"
                )
                before = list(client._client.call_tool_calls)
                with pytest.raises(ToolExecutionError):
                    await service.resume("timeout-thread", "approve")
                after = list(client._client.call_tool_calls)
                return started, before, after

    started, before, after = asyncio.run(scenario())
    assert started.status == "approval_required"
    assert before == []
    assert after == [("remote_action", {})]
