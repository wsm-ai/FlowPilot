import asyncio

import pytest

from app.retrieval.base import RetrievalError
from app.retrieval.in_memory import InMemoryRetriever
from app.retrieval.models import KnowledgeChunk, RetrievalQuery


def sample_chunks() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            chunk_id="CH-A",
            document_id="DOC-001",
            content="Login authentication fails for enterprise users",
            source="support-handbook",
            metadata={"department": "support"},
        ),
        KnowledgeChunk(
            chunk_id="CH-B",
            document_id="DOC-002",
            content="Billing invoices can be downloaded from account settings",
            source="billing-guide",
            metadata={"department": "finance"},
        ),
        KnowledgeChunk(
            chunk_id="CH-C",
            document_id="DOC-003",
            content="Authentication troubleshooting and login recovery",
            source="engineering-runbook",
            metadata={"department": "engineering"},
        ),
    ]


def search(retriever: InMemoryRetriever, **query_values):
    return asyncio.run(retriever.search(RetrievalQuery(**query_values)))


def test_search_returns_only_positive_matches_with_contiguous_ranks():
    results = search(
        InMemoryRetriever(sample_chunks()),
        query="login authentication",
    )

    assert [result.chunk.chunk_id for result in results] == ["CH-A", "CH-C"]
    assert all(result.score > 0 for result in results)
    assert all(0 <= result.score <= 1 for result in results)
    assert [result.rank for result in results] == [1, 2]


def test_top_k_limits_results():
    results = search(
        InMemoryRetriever(sample_chunks()),
        query="login authentication",
        top_k=1,
    )

    assert len(results) == 1


def test_equal_scores_are_ordered_by_chunk_id():
    chunks = [
        KnowledgeChunk(
            chunk_id=chunk_id,
            document_id=f"DOC-{chunk_id}",
            content="login guide",
            source="guide",
        )
        for chunk_id in ["CH-Z", "CH-A"]
    ]

    results = search(InMemoryRetriever(chunks), query="login")

    assert [result.chunk.chunk_id for result in results] == ["CH-A", "CH-Z"]


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        ({"department": "support"}, ["CH-A"]),
        ({"source": "billing-guide"}, ["CH-B"]),
        ({"document_id": "DOC-003"}, ["CH-C"]),
        ({"department": "legal"}, []),
    ],
)
def test_search_applies_exact_match_filters(filters, expected_ids):
    results = search(
        InMemoryRetriever(sample_chunks()),
        query="login authentication billing invoices",
        filters=filters,
    )

    assert [result.chunk.chunk_id for result in results] == expected_ids


def test_none_filter_only_matches_an_explicit_metadata_key():
    chunks = [
        KnowledgeChunk(
            chunk_id="CH-MISSING",
            document_id="DOC-001",
            content="login help",
            source="guide",
        ),
        KnowledgeChunk(
            chunk_id="CH-EXPLICIT",
            document_id="DOC-002",
            content="login help",
            source="guide",
            metadata={"department": None},
        ),
    ]

    results = search(
        InMemoryRetriever(chunks),
        query="login",
        filters={"department": None},
    )

    assert [result.chunk.chunk_id for result in results] == ["CH-EXPLICIT"]


def test_constructor_rejects_duplicate_chunk_ids():
    chunk = sample_chunks()[0]

    with pytest.raises(RetrievalError):
        InMemoryRetriever([chunk, chunk.model_copy(deep=True)])


def test_add_chunks_rejects_duplicates_atomically():
    existing = sample_chunks()[0]
    retriever = InMemoryRetriever([existing])
    valid_new = KnowledgeChunk(
        chunk_id="CH-NEW",
        document_id="DOC-NEW",
        content="unique searchable phrase",
        source="new-guide",
    )

    with pytest.raises(RetrievalError):
        retriever.add_chunks([valid_new, existing.model_copy(deep=True)])

    assert search(retriever, query="unique searchable phrase") == []


def test_input_container_and_models_are_defensively_copied():
    chunks = sample_chunks()
    retriever = InMemoryRetriever(chunks)
    chunks[0].content = "mutated external content"
    chunks.clear()

    results = search(retriever, query="login authentication")

    assert [result.chunk.chunk_id for result in results] == ["CH-A", "CH-C"]


def test_empty_corpus_returns_empty_results():
    assert search(InMemoryRetriever(), query="login") == []


def test_no_token_overlap_returns_empty_results():
    assert search(
        InMemoryRetriever(sample_chunks()),
        query="orchard telescope",
    ) == []
