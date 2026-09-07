import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from uuid import uuid4

import pytest
from mcp.types import CallToolResult, TextContent, Tool
from pydantic import ValidationError

import app.mcp.stdio_client as stdio_module
from app.mcp.client import (
    MCPConnectionError,
    MCPDiscoveryError,
    MCPToolCallError,
)
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.stdio_client import MCPStdioServerConfig, StdioMCPClient
from app.mcp.tool_adapter import MCPToolAdapter
from app.tools.base import ToolExecutionError


SERVER_PATH = (
    Path(__file__).parent / "fixtures" / "mcp_stdio_server.py"
).resolve()


def server_config() -> MCPStdioServerConfig:
    return MCPStdioServerConfig(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )


def test_stdio_config_validates_values_without_exposing_environment_in_repr():
    config = MCPStdioServerConfig(
        command=f" {sys.executable} ",
        args=[str(SERVER_PATH)],
        env={"SECRET_TOKEN": "private"},
        cwd=str(SERVER_PATH.parent),
    )
    assert config.command == sys.executable
    assert "SECRET_TOKEN" not in repr(config)
    assert "private" not in repr(config)

    with pytest.raises(ValidationError):
        MCPStdioServerConfig(command="   ")
    with pytest.raises(ValidationError):
        MCPStdioServerConfig(command=sys.executable, args=[1])
    with pytest.raises(ValidationError):
        MCPStdioServerConfig(command=sys.executable, env={"KEY": 1})


def test_disconnected_calls_fail_without_starting_transport():
    async def scenario():
        client = StdioMCPClient(server_config())
        with pytest.raises(
            MCPConnectionError,
            match="MCP client is not connected",
        ):
            await client.list_tools()
        with pytest.raises(
            MCPConnectionError,
            match="MCP client is not connected",
        ):
            await client.call_tool("add_numbers", {"a": 1, "b": 2})

    asyncio.run(scenario())


def test_connection_failure_uses_safe_error_message():
    command = f"flowpilot-missing-mcp-{uuid4().hex}.exe"

    async def scenario():
        with pytest.raises(MCPConnectionError) as raised:
            async with StdioMCPClient(
                MCPStdioServerConfig(command=command)
            ):
                pass
        return raised.value

    error = asyncio.run(scenario())
    assert str(error) == "MCP stdio connection failed"
    assert command not in str(error)


def test_real_stdio_discovery_call_and_sdk_normalization():
    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "add_numbers",
                {"a": 2, "b": 3},
            )
            return tools, result

    tools, result = asyncio.run(scenario())

    assert all(isinstance(tool, MCPRemoteTool) for tool in tools)
    assert [tool.name for tool in tools] == [
        "add_numbers",
        "echo_text",
        "fail_tool",
    ]
    assert isinstance(result, MCPToolResult)
    assert result.is_error is False
    assert result.structured_content == {"sum": 5}
    assert result.content
    assert all(isinstance(block, dict) for block in result.content)
    assert result.content[0]["type"] == "text"
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_adapter_executes_real_stdio_tool_and_maps_real_error_result():
    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            remote = next(
                tool
                for tool in await client.list_tools()
                if tool.name == "add_numbers"
            )
            add_adapter = MCPToolAdapter(
                client,
                local_name="mcp_test_add_numbers",
                remote_name=remote.name,
                description=remote.description,
                input_schema=remote.input_schema,
            )
            result = await add_adapter.execute({"a": 3, "b": 4})

            fail_remote = next(
                tool
                for tool in await client.list_tools()
                if tool.name == "fail_tool"
            )
            fail_adapter = MCPToolAdapter(
                client,
                local_name="mcp_test_fail_tool",
                remote_name=fail_remote.name,
                description=fail_remote.description,
                input_schema=fail_remote.input_schema,
            )
            with pytest.raises(ToolExecutionError) as raised:
                await fail_adapter.execute({})
            return result, raised.value

    result, error = asyncio.run(scenario())
    assert result == {"sum": 7}
    assert str(error) == "MCP tool execution failed"
    assert "sensitive server error" not in str(error)


class FakeSDKClient:
    pages = {}
    tool_result = CallToolResult(
        content=[TextContent(text="fallback")],
        structured_content={"ok": True},
        is_error=False,
    )

    def __init__(self, parameters):
        self.parameters = parameters
        self.cursors = []
        self.call_arguments = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def list_tools(self, *, cursor=None):
        self.cursors.append(cursor)
        return self.pages[cursor]

    async def call_tool(self, name, arguments=None):
        self.call_arguments = (name, arguments)
        return self.tool_result


def sdk_tool(name, schema=None):
    return Tool(
        name=name,
        description=f"Description for {name}",
        input_schema=schema or {"type": "object", "properties": {}},
    )


def test_paginated_discovery_preserves_page_order_and_cursors(monkeypatch):
    FakeSDKClient.pages = {
        None: SimpleNamespace(
            tools=[sdk_tool("tool_a"), sdk_tool("tool_b")],
            next_cursor="page-2",
        ),
        "page-2": SimpleNamespace(
            tools=[sdk_tool("tool_c")],
            next_cursor=None,
        ),
    }
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            tools = await client.list_tools()
            return tools, client._client.cursors

    tools, cursors = asyncio.run(scenario())
    assert [tool.name for tool in tools] == ["tool_a", "tool_b", "tool_c"]
    assert cursors == [None, "page-2"]


def test_duplicate_name_across_pages_is_rejected(monkeypatch):
    FakeSDKClient.pages = {
        None: SimpleNamespace(
            tools=[sdk_tool("search")],
            next_cursor="page-2",
        ),
        "page-2": SimpleNamespace(
            tools=[sdk_tool("search")],
            next_cursor=None,
        ),
    }
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            with pytest.raises(MCPDiscoveryError) as raised:
                await client.list_tools()
            return raised.value

    error = asyncio.run(scenario())
    assert str(error) == "MCP tool discovery returned duplicate tool name"


def test_invalid_discovery_data_maps_to_safe_error(monkeypatch):
    FakeSDKClient.pages = {
        None: SimpleNamespace(
            tools=[sdk_tool("bad", {"default": float("nan")})],
            next_cursor=None,
        )
    }
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            with pytest.raises(MCPDiscoveryError) as raised:
                await client.list_tools()
            return raised.value

    assert str(asyncio.run(scenario())) == (
        "MCP tool discovery returned invalid data"
    )


def test_fake_sdk_call_result_is_normalized_without_object_leakage(monkeypatch):
    FakeSDKClient.pages = {}
    monkeypatch.setattr(stdio_module, "Client", FakeSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            return await client.call_tool("test", {"value": 1})

    result = asyncio.run(scenario())
    assert result.structured_content == {"ok": True}
    assert result.content == [{"type": "text", "text": "fallback"}]
    assert isinstance(result.content[0], dict)


def test_sdk_tool_call_failure_maps_to_safe_domain_error(monkeypatch):
    class FailingSDKClient(FakeSDKClient):
        async def call_tool(self, name, arguments=None):
            raise OSError("sensitive transport detail")

    monkeypatch.setattr(stdio_module, "Client", FailingSDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            with pytest.raises(MCPToolCallError) as raised:
                await client.call_tool("test", {})
            return raised.value

    error = asyncio.run(scenario())
    assert str(error) == "MCP tool call failed"
    assert "sensitive" not in str(error)


def test_unexpected_sdk_runtime_error_is_not_swallowed(monkeypatch):
    class BuggySDKClient(FakeSDKClient):
        async def call_tool(self, name, arguments=None):
            raise RuntimeError("unexpected bug")

    monkeypatch.setattr(stdio_module, "Client", BuggySDKClient)

    async def scenario():
        async with StdioMCPClient(server_config()) as client:
            await client.call_tool("test", {})

    with pytest.raises(RuntimeError, match="unexpected bug"):
        asyncio.run(scenario())
