import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _validate_json_object(value: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-compatible") from exc
    return value


class KnowledgeDocument(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_id", "title", "source", "content")
    @classmethod
    def strings_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        return _validate_json_object(value)


class KnowledgeChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    position: int | None = Field(default=None, ge=0)

    @field_validator("chunk_id", "document_id", "content", "source")
    @classmethod
    def strings_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        return _validate_json_object(value)


class RetrievalQuery(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @field_validator("filters")
    @classmethod
    def filters_must_be_json_safe(
        cls,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        return _validate_json_object(value)


class RetrievalResult(BaseModel):
    chunk: KnowledgeChunk
    score: float
    rank: int | None = Field(default=None, ge=1)
