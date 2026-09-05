from typing import Any, Protocol

from pydantic import BaseModel

from app.retrieval.embeddings import EmbeddingVector
from app.retrieval.models import KnowledgeChunk, RetrievalResult


class VectorEntry(BaseModel):
    chunk: KnowledgeChunk
    embedding: EmbeddingVector


class VectorStoreError(Exception):
    """Raised when vector storage or comparison cannot complete safely."""


class VectorStore(Protocol):
    async def add(self, entries: list[VectorEntry]) -> None:
        ...

    async def search(
        self,
        embedding: EmbeddingVector,
        *,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        ...
