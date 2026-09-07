import json
import math

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.mcp.application import create_configured_mcp_client
from app.mcp.client import MCPConnectionError
from app.mcp.config import (
    MCPConfiguredStdioServerConfig,
    MCPHTTPServerConfig,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/mcp",
        "http://localhost:9000/mcp",
        "http://127.0.0.1:9000/mcp",
        "http://[::1]:9000/mcp",
    ],
)
def test_safe_http_urls_are_allowed(url):
    assert MCPHTTPServerConfig(
        server_id="test", transport="streamable_http", url=url
    ).url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/mcp",
        "file:///tmp/server",
        "ftp://example.com/mcp",
        "ws://example.com/mcp",
        "javascript:alert(1)",
        "data:text/plain,test",
        "https://user:password@example.com/mcp",
    ],
)
def test_unsafe_http_urls_are_rejected(url):
    with pytest.raises(ValidationError):
        MCPHTTPServerConfig(
            server_id="test", transport="streamable_http", url=url
        )


def test_explicit_insecure_remote_http_is_supported_for_controlled_development():
    config = MCPHTTPServerConfig(
        server_id="test",
        transport="streamable_http",
        url="http://example.com/mcp",
        allow_insecure_http=True,
    )
    assert config.allow_insecure_http is True


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["read_timeout_seconds", "connect_timeout_seconds"])
def test_http_timeouts_must_be_positive_and_finite(field, value):
    with pytest.raises(ValidationError):
        MCPHTTPServerConfig(
            server_id="test",
            transport="streamable_http",
            url="https://example.com/mcp",
            **{field: value},
        )


def test_settings_parse_discriminated_mcp_server_json(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {
                    "server_id": "local_tools",
                    "transport": "stdio",
                    "command": "python",
                },
                {
                    "server_id": "github",
                    "transport": "streamable_http",
                    "url": "https://example.com/mcp",
                },
            ]
        ),
    )
    settings = Settings(_env_file=None)
    assert isinstance(settings.mcp_servers[0], MCPConfiguredStdioServerConfig)
    assert isinstance(settings.mcp_servers[1], MCPHTTPServerConfig)


def test_empty_mcp_configuration_is_backward_compatible(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("MCP_SERVERS", raising=False)
    assert Settings(_env_file=None).mcp_servers == []


def test_missing_or_blank_auth_secret_fails_closed(monkeypatch):
    config = MCPHTTPServerConfig(
        server_id="github",
        transport="streamable_http",
        url="https://example.com/mcp",
        bearer_token_env="GITHUB_MCP_TOKEN",
    )
    monkeypatch.delenv("GITHUB_MCP_TOKEN", raising=False)
    with pytest.raises(
        MCPConnectionError,
        match="MCP authentication secret is not configured",
    ):
        create_configured_mcp_client(config)
    monkeypatch.setenv("GITHUB_MCP_TOKEN", "   ")
    with pytest.raises(MCPConnectionError):
        create_configured_mcp_client(config)


def test_resolved_secret_is_not_exposed(monkeypatch):
    token = "super-secret-token"
    monkeypatch.setenv("GITHUB_MCP_TOKEN", token)
    config = MCPHTTPServerConfig(
        server_id="github",
        transport="streamable_http",
        url="https://example.com/mcp",
        bearer_token_env="GITHUB_MCP_TOKEN",
    )
    client = create_configured_mcp_client(config)
    assert token not in repr(config)
    assert token not in repr(client)


def test_unknown_transport_and_invalid_server_id_fail_validation(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv(
        "MCP_SERVERS",
        '[{"server_id":"bad-id","transport":"unknown"}]',
    )
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
