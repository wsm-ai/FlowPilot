from copy import deepcopy
from typing import Protocol

from app.retrieval.models import KnowledgeChunk, KnowledgeDocument


class ChunkingError(Exception):
    """Raised when a document cannot be split into knowledge chunks."""


class DocumentChunker(Protocol):
    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        ...


class SimpleTextChunker:
    """Deterministic character-window chunker with optional overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        try:
            content = document.content
            step = self._chunk_size - self._chunk_overlap
            chunks: list[KnowledgeChunk] = []
            start = 0
            position = 0
            while start < len(content):
                chunk_content = content[start : start + self._chunk_size]
                metadata = deepcopy(document.metadata)
                metadata["document_title"] = document.title
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{document.document_id}:chunk:{position}",
                        document_id=document.document_id,
                        content=chunk_content,
                        source=document.source,
                        metadata=metadata,
                        position=position,
                    )
                )
                if start + self._chunk_size >= len(content):
                    break
                start += step
                position += 1
            return chunks
        except (AttributeError, TypeError, ValueError) as exc:
            raise ChunkingError("Unable to chunk document") from exc
