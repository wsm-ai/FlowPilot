from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.mcp.naming import is_valid_server_id
from app.mcp.stdio_client import MCPStdioServerConfig


class _MCPApplicationServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    server_id: str

    @field_validator("server_id")
    @classmethod
    def validate_server_id(cls, value: str) -> str:
        if not is_valid_server_id(value):
            raise ValueError("Invalid MCP server ID")
        return value


class MCPConfiguredStdioServerConfig(
    MCPStdioServerConfig,
    _MCPApplicationServerConfig,
):
    transport: Literal["stdio"]


class MCPHTTPServerConfig(_MCPApplicationServerConfig):
    transport: Literal["streamable_http"]
    url: str
    read_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    connect_timeout_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    bearer_token_env: str | None = None
    allow_insecure_http: bool = False

    @field_validator("bearer_token_env")
    @classmethod
    def token_environment_name_must_not_be_blank(
        cls, value: str | None
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("bearer_token_env must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_url_security(self) -> "MCPHTTPServerConfig":
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("MCP HTTP URL must use HTTP or HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("MCP HTTP URL must not contain credentials")
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if (
            parsed.scheme == "http"
            and parsed.hostname not in local_hosts
            and not self.allow_insecure_http
        ):
            raise ValueError("Insecure remote MCP HTTP URL is not allowed")
        return self


MCPServerConfig = Annotated[
    MCPConfiguredStdioServerConfig | MCPHTTPServerConfig,
    Field(discriminator="transport"),
]
