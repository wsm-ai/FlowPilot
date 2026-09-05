import math
from typing import Protocol

from pydantic import BaseModel, Field, field_validator


class EmbeddingError(Exception):
    """Raised when embedding infrastructure cannot complete an operation."""


class EmbeddingVector(BaseModel):
    values: list[float] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def values_must_be_finite(cls, values: list[float]) -> list[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("embedding values must be finite")
        return values


class EmbeddingProvider(Protocol):
    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        ...

    async def embed_query(self, text: str) -> EmbeddingVector:
        ...
