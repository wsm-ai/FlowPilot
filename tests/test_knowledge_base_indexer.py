import asyncio

import pytest

from app.retrieval.chunking import ChunkingError, SimpleTextChunker
from app.retrieval.hash_embedding import HashEmbeddingProvider
from app.retrieval.in_memory_vector_store import InMemoryVectorStore
from app.retrieval.indexing import KnowledgeBaseIndexer
from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.models import KnowledgeDocument, RetrievalQuery
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.vector_store import VectorStoreError


def document() -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id="DOC-001",
        title="Support Handbook",
        source="support-handbook",
        content="Login authentication fails for enterprise users",
    )


def test_index_document_writes_chunks_searchable_by_identical_text():
    provider = HashEmbeddingProvider(dimensions=16)
    store = InMemoryVectorStore()
    ingestion = DocumentIngestionService(SimpleTextChunker(), provider)
    indexer = KnowledgeBaseIndexer(ingestion, store)

    chunks = asyncio.run(indexer.index_document(document()))
    results = asyncio.run(
        VectorRetriever(provider, store).search(
            RetrievalQuery(query=chunks[0].content)
        )
    )

    assert [chunk.position for chunk in chunks] == [0]
    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    assert results[0].score == pytest.approx(1.0)


def test_ingestion_failure_does_not_call_vector_store():
    class FailingIngestion:
        async def ingest(self, source_document):
            raise ChunkingError("chunking failed")

    class SpyStore:
        called = False

        async def add(self, entries):
            self.called = True

    store = SpyStore()
    indexer = KnowledgeBaseIndexer(FailingIngestion(), store)

    with pytest.raises(ChunkingError):
        asyncio.run(indexer.index_document(document()))
    assert store.called is False


def test_vector_store_failure_is_propagated_without_partial_result():
    class FailingStore:
        async def add(self, entries):
            raise VectorStoreError("store failed")

    ingestion = DocumentIngestionService(
        SimpleTextChunker(), HashEmbeddingProvider()
    )
    indexer = KnowledgeBaseIndexer(ingestion, FailingStore())

    with pytest.raises(VectorStoreError, match="store failed"):
        asyncio.run(indexer.index_document(document()))


def test_empty_ingestion_returns_without_store_add():
    class EmptyIngestion:
        async def ingest(self, source_document):
            return []

    class SpyStore:
        called = False

        async def add(self, entries):
            self.called = True

    store = SpyStore()
    result = asyncio.run(KnowledgeBaseIndexer(EmptyIngestion(), store).index_document(document()))

    assert result == []
    assert store.called is False
