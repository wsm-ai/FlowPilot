from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.mcp.config import MCPServerConfig
from app.security.rbac import APIKeyRoleBinding


class Settings(BaseSettings):
    deepseek_api_key: SecretStr
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    flowpilot_auth_enabled: bool = True
    flowpilot_api_key: SecretStr | None = None
    flowpilot_api_keys: list[APIKeyRoleBinding] = Field(default_factory=list)
    flowpilot_max_request_body_bytes: int = Field(
        default=1_048_576,
        gt=0,
        le=16_777_216,
    )

    @field_validator("flowpilot_api_key")
    @classmethod
    def normalize_flowpilot_api_key(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        if value is None:
            return None
        normalized = value.get_secret_value().strip()
        return SecretStr(normalized) if normalized else None

    @model_validator(mode="after")
    def require_flowpilot_api_key_when_authentication_is_enabled(
        self,
    ) -> "Settings":
        if (
            self.flowpilot_auth_enabled
            and self.flowpilot_api_key is None
            and not self.flowpilot_api_keys
        ):
            raise ValueError(
                "FLOWPILOT_API_KEY is required when authentication is enabled"
            )
        configured_keys = [
            binding.key.get_secret_value() for binding in self.flowpilot_api_keys
        ]
        if self.flowpilot_api_key is not None:
            configured_keys.append(self.flowpilot_api_key.get_secret_value())
        if len(configured_keys) != len(set(configured_keys)):
            raise ValueError("FlowPilot API keys must be unique")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
