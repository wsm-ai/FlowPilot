import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _non_blank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be blank")
    return normalized


class GroundingEvidence(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    content: str
    source: str
    score: float
    rank: int | None = Field(default=None, ge=1)
    position: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("citation_id", "chunk_id", "document_id", "content", "source")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-compatible") from exc
        return value


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    title: str | None = None

    @field_validator("citation_id", "chunk_id", "document_id", "source")
    @classmethod
    def required_strings_must_not_be_blank(cls, value: str) -> str:
        return _non_blank(value)


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        return _non_blank(value)


class GroundedSynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citation_ids: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        return _non_blank(value)

    @field_validator("citation_ids")
    @classmethod
    def citation_ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("citation IDs must not be blank")
        return values
