import asyncio

import pytest

from app.retrieval.embeddings import EmbeddingError, EmbeddingVector
from app.retrieval.models import KnowledgeChunk, RetrievalQuery, RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.vector_store import VectorStoreError


class FakeEmbeddingProvider:
    def __init__(self, error=None) -> None:
        self.queries = []
        self.error = error

    async def embed_query(self, text):
        self.queries.append(text)
        if self.error:
            raise self.error
        return EmbeddingVector(values=[1.0, 0.0])


class FakeVectorStore:
    def __init__(self, results=None, error=None) -> None:
        self.calls = []
        self.results = results or []
        self.error = error

    async def search(self, embedding, *, top_k, filters):
        self.calls.append((embedding, top_k, filters))
        if self.error:
            raise self.error
        return self.results


def result() -> RetrievalResult:
    return RetrievalResult(
        chunk=KnowledgeChunk(
            chunk_id="CH-A",
            document_id="DOC-A",
            content="enterprise login",
            source="handbook",
        ),
        score=1.0,
        rank=1,
    )


def test_vector_retriever_forwards_embedding_query_and_search_options():
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore([result()])
    retriever = VectorRetriever(provider, store)

    results = asyncio.run(
        retriever.search(
            RetrievalQuery(
                query="enterprise login",
                top_k=3,
                filters={"department": "support"},
            )
        )
    )

    assert provider.queries == ["enterprise login"]
    embedding, top_k, filters = store.calls[0]
    assert embedding == EmbeddingVector(values=[1.0, 0.0])
    assert top_k == 3
    assert filters == {"department": "support"}
    assert results == [result()]


def test_embedding_failure_is_propagated_without_store_call():
    error = EmbeddingError("embedding unavailable")
    provider = FakeEmbeddingProvider(error=error)
    store = FakeVectorStore()

    with pytest.raises(EmbeddingError) as raised:
        asyncio.run(VectorRetriever(provider, store).search(RetrievalQuery(query="login")))

    assert raised.value is error
    assert store.calls == []


def test_vector_store_failure_is_propagated():
    error = VectorStoreError("store unavailable")
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore(error=error)

    with pytest.raises(VectorStoreError) as raised:
        asyncio.run(VectorRetriever(provider, store).search(RetrievalQuery(query="login")))

    assert raised.value is error
