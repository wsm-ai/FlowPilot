import asyncio
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest
from pydantic import SecretStr

import app.mcp.http_client as http_module

from app.mcp.config import MCPHTTPServerConfig
from app.mcp.http_client import StreamableHTTPMCPClient
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.tool_composition import MCPServerClientBinding, compose_mcp_tools
from app.tools.registry import ToolRegistry


SERVER_PATH = (
    Path(__file__).parent / "fixtures" / "mcp_http_server.py"
).resolve()


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def running_http_mcp_server():
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(SERVER_PATH), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("MCP HTTP test server exited during startup")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("MCP HTTP test server did not become ready")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def test_real_streamable_http_discovery_call_and_registry_composition():
    async def scenario(url):
        config = MCPHTTPServerConfig(
            server_id="test",
            transport="streamable_http",
            url=url,
        )
        async with StreamableHTTPMCPClient(config) as client:
            tools = await client.list_tools()
            direct = await client.call_tool("add_numbers", {"a": 2, "b": 3})
            registry = ToolRegistry()
            composition = await compose_mcp_tools(
                registry,
                [MCPServerClientBinding("test", client)],
            )
            composed = await registry.execute(
                "mcp_test_add_numbers", {"a": 4, "b": 5}
            )
            return tools, direct, composition, composed

    with running_http_mcp_server() as url:
        tools, direct, composition, composed = asyncio.run(scenario(url))

    assert all(isinstance(tool, MCPRemoteTool) for tool in tools)
    assert {tool.name for tool in tools} == {"add_numbers", "echo_text"}
    assert isinstance(direct, MCPToolResult)
    assert direct.structured_content == {"sum": 5}
    assert "mcp_test_add_numbers" in composition.local_names
    assert composed == {"sum": 9}


def test_http_client_builds_official_transport_with_auth_and_timeouts(
    monkeypatch,
):
    captured = {}

    class FakeHTTPClient:
        def __init__(self, **kwargs):
            captured["http_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class FakeSDKClient:
        def __init__(self, transport, *, read_timeout_seconds=None):
            captured["transport"] = transport
            captured["sdk_timeout"] = read_timeout_seconds

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    marker = object()

    def fake_transport(url, *, http_client):
        captured["url"] = url
        captured["transport_http_client"] = http_client
        return marker

    monkeypatch.setattr(http_module.httpx2, "AsyncClient", FakeHTTPClient)
    monkeypatch.setattr(http_module, "streamable_http_client", fake_transport)
    monkeypatch.setattr(http_module, "Client", FakeSDKClient)
    config = MCPHTTPServerConfig(
        server_id="github",
        transport="streamable_http",
        url="https://example.com/mcp",
        read_timeout_seconds=12,
        connect_timeout_seconds=3,
    )

    async def scenario():
        async with StreamableHTTPMCPClient(
            config,
            bearer_token=SecretStr("super-secret-token"),
        ):
            pass

    asyncio.run(scenario())
    assert captured["transport"] is marker
    assert captured["url"] == config.url
    assert captured["sdk_timeout"] == 12
    assert captured["http_kwargs"]["headers"] == {
        "Authorization": "Bearer super-secret-token"
    }
    assert "follow_redirects" not in captured["http_kwargs"]
    timeout = captured["http_kwargs"]["timeout"]
    assert timeout.connect == 3
    assert timeout.read == 12
