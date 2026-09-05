import asyncio

import pytest

from app.retrieval.chunking import ChunkingError, SimpleTextChunker
from app.retrieval.embeddings import EmbeddingError, EmbeddingVector
from app.retrieval.hash_embedding import HashEmbeddingProvider
from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.models import KnowledgeDocument


def document() -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id="DOC-001",
        title="Support Handbook",
        source="support-handbook",
        content="ABCDEFGHIJKLMNO",
    )


def test_ingestion_preserves_chunk_order_and_embedding_dimensions():
    service = DocumentIngestionService(
        SimpleTextChunker(chunk_size=6, chunk_overlap=2),
        HashEmbeddingProvider(dimensions=12),
    )

    results = asyncio.run(service.ingest(document()))

    assert len(results) == 4
    assert [item.chunk.position for item in results] == [0, 1, 2, 3]
    assert [item.chunk.content for item in results] == [
        "ABCDEF",
        "EFGHIJ",
        "IJKLMN",
        "MNO",
    ]
    assert all(len(item.embedding.values) == 12 for item in results)


def test_ingestion_rejects_embedding_count_mismatch():
    class BrokenProvider:
        async def embed_documents(self, texts):
            return [EmbeddingVector(values=[1.0])] * (len(texts) - 1)

    service = DocumentIngestionService(
        SimpleTextChunker(chunk_size=6, chunk_overlap=2),
        BrokenProvider(),
    )

    with pytest.raises(EmbeddingError):
        asyncio.run(service.ingest(document()))


def test_ingestion_propagates_embedding_failure():
    class FailingProvider:
        async def embed_documents(self, texts):
            raise EmbeddingError("embedding failed")

    service = DocumentIngestionService(SimpleTextChunker(), FailingProvider())

    with pytest.raises(EmbeddingError, match="embedding failed"):
        asyncio.run(service.ingest(document()))


def test_ingestion_propagates_chunking_failure_without_embedding_call():
    class FailingChunker:
        def chunk(self, source_document):
            raise ChunkingError("chunking failed")

    class SpyProvider:
        called = False

        async def embed_documents(self, texts):
            self.called = True
            return []

    provider = SpyProvider()
    service = DocumentIngestionService(FailingChunker(), provider)

    with pytest.raises(ChunkingError, match="chunking failed"):
        asyncio.run(service.ingest(document()))
    assert provider.called is False


def test_ingestion_returns_empty_without_embedding_empty_chunks():
    class EmptyChunker:
        def chunk(self, source_document):
            return []

    class SpyProvider:
        called = False

        async def embed_documents(self, texts):
            self.called = True
            return []

    provider = SpyProvider()
    service = DocumentIngestionService(EmptyChunker(), provider)

    assert asyncio.run(service.ingest(document())) == []
    assert provider.called is False
