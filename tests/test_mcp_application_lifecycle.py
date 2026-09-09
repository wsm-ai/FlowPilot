import asyncio
from contextlib import AsyncExitStack
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app.api.agent as agent_api
import app.main as main_module
import app.mcp.application as application_module
from app.api.dependencies import (
    get_mcp_approval_required_actions,
    get_tool_registry,
)
from app.core.config import Settings
from app.mcp.application import compose_configured_mcp_tools
from app.mcp.client import (
    MCPAuthenticationError,
    MCPConnectionConfigurationError,
    MCPConnectionError,
    MCPPermanentConnectionError,
    MCPTransientConnectionError,
)
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.tools.registry import ToolRegistry


HTTP_SERVER_PATH = (
    Path(__file__).parent / "fixtures" / "mcp_http_server.py"
).resolve()


@contextmanager
def running_http_server():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, str(HTTP_SERVER_PATH), str(port)],
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


class LifecycleClient:
    def __init__(self, name, events, *, enter_error=None):
        self.name = name
        self.events = events
        self.enter_error = enter_error

    async def __aenter__(self):
        self.events.append(f"enter:{self.name}")
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.events.append(f"exit:{self.name}")

    async def list_tools(self):
        return [
            MCPRemoteTool(
                name="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ]

    async def call_tool(self, name, arguments):
        return MCPToolResult(structured_content={"server": self.name})


def config(server_id):
    return SimpleNamespace(server_id=server_id, required=True)


def test_multiple_transport_clients_share_registry_and_close_in_reverse_order(
    monkeypatch,
):
    events = []
    clients = {
        "stdio": LifecycleClient("stdio", events),
        "http": LifecycleClient("http", events),
    }
    monkeypatch.setattr(
        application_module,
        "create_configured_mcp_client",
        lambda item: clients[item.server_id],
    )

    async def scenario():
        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            composition = await compose_configured_mcp_tools(
                registry,
                [config("stdio"), config("http")],
                stack,
            )
            names_during_lifespan = {
                definition["function"]["name"]
                for definition in registry.definitions()
            }
        return composition, names_during_lifespan

    composition, names = asyncio.run(scenario())
    assert names == {"mcp_stdio_search", "mcp_http_search"}
    assert composition.approval_required_actions == names
    assert events == ["enter:stdio", "enter:http", "exit:http", "exit:stdio"]


def test_second_server_startup_failure_closes_first_and_registers_nothing(
    monkeypatch,
):
    events = []
    clients = {
        "server_a": LifecycleClient("server_a", events),
        "server_b": LifecycleClient(
            "server_b",
            events,
            enter_error=MCPConnectionError("safe startup failure"),
        ),
    }
    monkeypatch.setattr(
        application_module,
        "create_configured_mcp_client",
        lambda item: clients[item.server_id],
    )

    async def scenario():
        registry = ToolRegistry()
        with pytest.raises(MCPConnectionError, match="safe startup failure"):
            async with AsyncExitStack() as stack:
                await compose_configured_mcp_tools(
                    registry,
                    [config("server_a"), config("server_b")],
                    stack,
                )
        return registry.definitions()

    assert asyncio.run(scenario()) == []
    assert events == ["enter:server_a", "enter:server_b", "exit:server_a"]


def test_production_lifespan_composes_http_tools_and_planner_policy(monkeypatch):
    with running_http_server() as url:
        settings = Settings(
            _env_file=None,
            deepseek_api_key="test-key",
            mcp_servers=[
                {
                    "server_id": "test",
                    "transport": "streamable_http",
                    "url": url,
                }
            ],
        )
        monkeypatch.setattr(main_module, "get_settings", lambda: settings)

        with TestClient(main_module.app):
            request = SimpleNamespace(app=main_module.app)
            registry = get_tool_registry(request)
            policy = get_mcp_approval_required_actions(request)
            names = {
                definition["function"]["name"]
                for definition in registry.definitions()
            }
            service = agent_api.get_approval_workflow_service(
                llm_service=object(),
                checkpointer=main_module.app.state.checkpointer,
                registry=registry,
                approval_required_actions=policy,
                side_effect_executor=main_module.app.state.side_effect_executor,
            )

    assert names == {
        "get_customer_feedback",
        "search_knowledge_base",
        "mcp_test_add_numbers",
        "mcp_test_echo_text",
    }
    assert policy == {"mcp_test_add_numbers", "mcp_test_echo_text"}
    assert service._planner_service._approval_required_actions == policy


def test_optional_unavailable_mcp_server_allows_degraded_app_startup(
    monkeypatch,
):
    events = []
    unavailable = LifecycleClient(
        "optional",
        events,
        enter_error=MCPTransientConnectionError("TOKEN=private endpoint"),
    )
    monkeypatch.setattr(
        application_module,
        "create_configured_mcp_client",
        lambda config: unavailable,
    )
    settings = Settings(
        _env_file=None,
        deepseek_api_key="test-key",
        mcp_servers=[
            {
                "server_id": "optional",
                "transport": "stdio",
                "command": "unused",
                "required": False,
            }
        ],
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)

    with TestClient(main_module.app) as client:
        health_response = client.get("/health")
        names = {
            item["function"]["name"]
            for item in main_module.app.state.tool_registry.definitions()
        }
        degradations = main_module.app.state.mcp_degradations

    assert names == {"get_customer_feedback", "search_knowledge_base"}
    assert health_response.status_code == 200
    assert main_module.app.state.mcp_approval_required_actions == frozenset()
    assert len(degradations) == 1
    assert degradations[0].component == "optional"
    assert degradations[0].safe_message == (
        "MCP connection temporarily unavailable"
    )
    assert "private" not in repr(degradations[0])


@pytest.mark.parametrize(
    "error",
    [
        MCPAuthenticationError("TOKEN=private"),
        MCPConnectionConfigurationError("command=private"),
        MCPPermanentConnectionError("response=private"),
        MCPConnectionError("unknown=private"),
    ],
)
def test_optional_non_transient_connection_failure_is_fail_closed(
    monkeypatch, error
):
    client = LifecycleClient("optional", [], enter_error=error)
    monkeypatch.setattr(
        application_module,
        "create_configured_mcp_client",
        lambda item: client,
    )
    registry = ToolRegistry()
    optional_config = SimpleNamespace(server_id="optional", required=False)

    async def scenario():
        async with AsyncExitStack() as stack:
            with pytest.raises(type(error)) as raised:
                await compose_configured_mcp_tools(
                    registry, [optional_config], stack
                )
            return raised.value

    raised = asyncio.run(scenario())
    assert raised is error
    assert registry.definitions() == []
