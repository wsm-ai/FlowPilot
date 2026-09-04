from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentRunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("thread_id")
    @classmethod
    def thread_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("thread_id must not be blank")
        return value


class ExecutedToolResponse(BaseModel):
    tool_call_id: str
    name: str
    arguments: dict[str, Any]


class AgentRunResponse(BaseModel):
    run_id: str
    answer: str
    executed_tools: list[ExecutedToolResponse]
