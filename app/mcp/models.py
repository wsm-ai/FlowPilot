import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-compatible") from exc
    return value


class MCPRemoteTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any]

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("input_schema")
    @classmethod
    def input_schema_must_be_json_safe(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        return _validate_json_safe(value)


class MCPToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: list[Any] = Field(default_factory=list)
    structured_content: dict[str, Any] | None = None
    is_error: bool = False

    @field_validator("content")
    @classmethod
    def content_must_be_json_safe(cls, value: list[Any]) -> list[Any]:
        return _validate_json_safe(value)

    @field_validator("structured_content")
    @classmethod
    def structured_content_must_be_json_safe(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        return _validate_json_safe(value)
