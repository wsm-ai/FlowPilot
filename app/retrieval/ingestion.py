from dataclasses import dataclass

from app.retrieval.chunking import DocumentChunker
from app.retrieval.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingVector,
)
from app.retrieval.models import KnowledgeChunk, KnowledgeDocument


@dataclass(slots=True)
class IngestedChunk:
    chunk: KnowledgeChunk
    embedding: EmbeddingVector


class DocumentIngestionService:
    def __init__(
        self,
        chunker: DocumentChunker,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._chunker = chunker
        self._embedding_provider = embedding_provider

    async def ingest(
        self,
        document: KnowledgeDocument,
    ) -> list[IngestedChunk]:
        chunks = self._chunker.chunk(document)
        if not chunks:
            return []

        embeddings = await self._embedding_provider.embed_documents(
            [chunk.content for chunk in chunks]
        )
        if len(embeddings) != len(chunks):
            raise EmbeddingError(
                "Embedding count does not match knowledge chunk count"
            )

        return [
            IngestedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
