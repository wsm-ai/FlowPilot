import asyncio
import json

import pytest

from app.mcp.client import (
    MCPConnectionError,
    MCPDiscoveryError,
    MCPToolCallError,
)
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.tool_adapter import MCPToolAdapter
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry


def input_schema():
    return {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


class FakeMCPClient:
    def __init__(self, result=None, error=None, *, mutate_arguments=False):
        self.result = result or MCPToolResult(content=[])
        self.error = error
        self.mutate_arguments = mutate_arguments
        self.list_tools_calls = 0
        self.call_tool_calls = []

    async def list_tools(self):
        self.list_tools_calls += 1
        return [
            MCPRemoteTool(
                name="search_issues",
                description="Search issues",
                input_schema=input_schema(),
            )
        ]

    async def call_tool(self, name, arguments):
        self.call_tool_calls.append((name, arguments))
        if self.mutate_arguments:
            arguments["filters"]["state"] = "mutated"
        if self.error is not None:
            raise self.error
        return self.result


def adapter(client):
    return MCPToolAdapter(
        client,
        local_name="mcp_github_search_issues",
        remote_name="search_issues",
        description="Search remote issues",
        input_schema=input_schema(),
    )


def test_adapter_separates_local_registry_name_from_remote_call_name():
    client = FakeMCPClient(
        MCPToolResult(structured_content={"issues": [{"id": 1}]})
    )
    tool = adapter(client)
    registry = ToolRegistry()
    registry.register(tool)

    result = asyncio.run(
        registry.execute(
            "mcp_github_search_issues",
            {"query": "login"},
        )
    )

    assert tool.name == "mcp_github_search_issues"
    assert client.call_tool_calls == [("search_issues", {"query": "login"})]
    assert result == {"issues": [{"id": 1}]}
    assert registry.definitions()[0]["function"]["name"] == tool.name
    json.dumps(result, allow_nan=False)


def test_parameters_are_defensively_copied():
    tool = adapter(FakeMCPClient())
    exposed = tool.parameters
    exposed["properties"].clear()
    assert tool.parameters == input_schema()


def test_client_cannot_mutate_caller_arguments():
    client = FakeMCPClient(mutate_arguments=True)
    arguments = {"query": "login", "filters": {"state": "open"}}

    asyncio.run(adapter(client).execute(arguments))

    assert arguments == {"query": "login", "filters": {"state": "open"}}
    assert client.call_tool_calls[0][1]["filters"]["state"] == "mutated"


def test_content_is_returned_without_guessing_text_as_json():
    content = [{"type": "text", "text": '{"looks":"like json"}'}]
    result = asyncio.run(
        adapter(FakeMCPClient(MCPToolResult(content=content))).execute({})
    )
    assert result == content
    assert isinstance(result, list)
    json.dumps(result, allow_nan=False)


def test_mcp_error_result_maps_to_safe_tool_error():
    tool = adapter(
        FakeMCPClient(
            MCPToolResult(
                content=[{"type": "text", "text": "sensitive server error"}],
                is_error=True,
            )
        )
    )
    with pytest.raises(ToolExecutionError) as raised:
        asyncio.run(tool.execute({"query": "login"}))
    assert str(raised.value) == "MCP tool execution failed"
    assert "sensitive" not in str(raised.value)


@pytest.mark.parametrize(
    "error",
    [
        MCPConnectionError("secret command and environment"),
        MCPDiscoveryError("private discovery response"),
        MCPToolCallError("sensitive server body"),
    ],
)
def test_known_mcp_errors_map_to_safe_tool_error(error):
    with pytest.raises(ToolExecutionError) as raised:
        asyncio.run(adapter(FakeMCPClient(error=error)).execute({}))
    assert str(raised.value) == "MCP tool execution failed"
    assert raised.value.__cause__ is error
    assert str(error) not in str(raised.value)


def test_unexpected_programming_error_is_not_swallowed():
    error = RuntimeError("unexpected bug")
    with pytest.raises(RuntimeError, match="unexpected bug") as raised:
        asyncio.run(adapter(FakeMCPClient(error=error)).execute({}))
    assert raised.value is error


def test_invalid_json_arguments_are_rejected_before_client_call():
    client = FakeMCPClient()
    with pytest.raises(ToolExecutionError, match="Invalid MCP tool arguments"):
        asyncio.run(adapter(client).execute({"value": float("nan")}))
    assert client.call_tool_calls == []
