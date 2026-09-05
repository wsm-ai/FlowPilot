import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.chunking import SimpleTextChunker
from app.retrieval.demo_knowledge import (
    DEMO_KNOWLEDGE_DOCUMENTS,
    bootstrap_demo_knowledge_base,
)
from app.retrieval.hash_embedding import HashEmbeddingProvider
from app.retrieval.in_memory_vector_store import InMemoryVectorStore
from app.retrieval.indexing import KnowledgeBaseIndexer
from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.models import RetrievalQuery
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.vector_store import VectorStoreError
from app.tools.registry import create_default_tool_registry


def test_demo_documents_are_stable_unique_and_json_safe():
    assert len(DEMO_KNOWLEDGE_DOCUMENTS) == 3
    document_ids = [document.document_id for document in DEMO_KNOWLEDGE_DOCUMENTS]
    assert len(document_ids) == len(set(document_ids))
    assert all(document.source for document in DEMO_KNOWLEDGE_DOCUMENTS)
    for document in DEMO_KNOWLEDGE_DOCUMENTS:
        json.dumps(document.metadata, allow_nan=False)


def test_real_bootstrap_indexes_documents_into_shared_search_store():
    async def scenario():
        embedding_provider = HashEmbeddingProvider(dimensions=16)
        vector_store = InMemoryVectorStore()
        ingestion = DocumentIngestionService(
            SimpleTextChunker(),
            embedding_provider,
        )
        indexer = KnowledgeBaseIndexer(ingestion, vector_store)
        chunks = await bootstrap_demo_knowledge_base(indexer)
        retriever = VectorRetriever(embedding_provider, vector_store)
        results = await retriever.search(
            RetrievalQuery(query=chunks[0].content, top_k=1)
        )
        return chunks, results

    chunks, results = asyncio.run(scenario())

    assert chunks
    assert results
    assert results[0].chunk.document_id == chunks[0].document_id
    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    assert results[0].score == pytest.approx(1.0)


def test_bootstrap_propagates_indexing_failure():
    error = VectorStoreError("demo indexing failed")

    class FailingIndexer:
        async def index_document(self, document):
            raise error

    with pytest.raises(VectorStoreError) as raised:
        asyncio.run(bootstrap_demo_knowledge_base(FailingIndexer()))

    assert raised.value is error


def test_application_lifespan_exposes_initialized_retriever_only():
    with TestClient(app):
        assert hasattr(app.state, "retriever")
        assert not hasattr(app.state, "indexer")
        assert not hasattr(app.state, "vector_store")
        exact_text = DEMO_KNOWLEDGE_DOCUMENTS[0].content
        results = asyncio.run(
            app.state.retriever.search(
                RetrievalQuery(query=exact_text, top_k=1)
            )
        )

    assert results[0].chunk.document_id == "DOC-001"
    assert results[0].score == pytest.approx(1.0)


def test_default_registry_does_not_register_knowledge_base_tool():
    names = [
        definition["function"]["name"]
        for definition in create_default_tool_registry().definitions()
    ]

    assert names == ["get_customer_feedback"]
