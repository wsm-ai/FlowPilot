import math
from typing import Any

from app.retrieval.embeddings import EmbeddingVector
from app.retrieval.models import KnowledgeChunk, RetrievalResult
from app.retrieval.vector_store import VectorEntry, VectorStoreError


def _cosine_similarity(
    left: EmbeddingVector,
    right: EmbeddingVector,
) -> float:
    if len(left.values) != len(right.values):
        raise VectorStoreError("Embedding dimensions do not match")

    dot_product = sum(a * b for a, b in zip(left.values, right.values))
    left_norm = math.sqrt(sum(value * value for value in left.values))
    right_norm = math.sqrt(sum(value * value for value in right.values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._entries: dict[str, VectorEntry] = {}
        self._dimensions: int | None = None

    async def add(self, entries: list[VectorEntry]) -> None:
        if not entries:
            return

        chunk_ids = [entry.chunk.chunk_id for entry in entries]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise VectorStoreError("Duplicate chunk ID in vector batch")
        if any(chunk_id in self._entries for chunk_id in chunk_ids):
            raise VectorStoreError("Vector entry already exists")

        dimensions = {len(entry.embedding.values) for entry in entries}
        if len(dimensions) != 1:
            raise VectorStoreError("Vector batch has mixed dimensions")
        batch_dimensions = dimensions.pop()
        if self._dimensions is not None and batch_dimensions != self._dimensions:
            raise VectorStoreError("Embedding dimension does not match vector store")

        additions = {
            entry.chunk.chunk_id: entry.model_copy(deep=True) for entry in entries
        }
        self._entries.update(additions)
        if self._dimensions is None:
            self._dimensions = batch_dimensions

    async def search(
        self,
        embedding: EmbeddingVector,
        *,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1")
        if not self._entries:
            return []
        if len(embedding.values) != self._dimensions:
            raise VectorStoreError("Query embedding dimension does not match store")

        matches: list[tuple[float, KnowledgeChunk]] = []
        for entry in self._entries.values():
            if not self._matches_filters(entry.chunk, filters):
                continue
            score = _cosine_similarity(embedding, entry.embedding)
            matches.append((score, entry.chunk))

        matches.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalResult(
                chunk=chunk.model_copy(deep=True),
                score=score,
                rank=rank,
            )
            for rank, (score, chunk) in enumerate(matches[:top_k], start=1)
        ]

    @staticmethod
    def _matches_filters(
        chunk: KnowledgeChunk,
        filters: dict[str, Any],
    ) -> bool:
        for name, expected in filters.items():
            if name == "document_id":
                actual = chunk.document_id
            elif name == "source":
                actual = chunk.source
            else:
                if name not in chunk.metadata:
                    return False
                actual = chunk.metadata[name]
            if actual != expected:
                return False
        return True
