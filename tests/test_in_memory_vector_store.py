import asyncio

import pytest

from app.retrieval.embeddings import EmbeddingVector
from app.retrieval.in_memory_vector_store import InMemoryVectorStore
from app.retrieval.models import KnowledgeChunk
from app.retrieval.vector_store import VectorEntry, VectorStoreError


def entry(
    chunk_id: str,
    values: list[float],
    *,
    document_id: str | None = None,
    source: str = "handbook",
    metadata: dict | None = None,
) -> VectorEntry:
    return VectorEntry(
        chunk=KnowledgeChunk(
            chunk_id=chunk_id,
            document_id=document_id or f"DOC-{chunk_id}",
            content=f"Content for {chunk_id}",
            source=source,
            metadata=metadata or {},
        ),
        embedding=EmbeddingVector(values=values),
    )


def run(coroutine):
    return asyncio.run(coroutine)


def search(store, values, *, top_k=10, filters=None):
    return run(
        store.search(
            EmbeddingVector(values=values),
            top_k=top_k,
            filters=filters or {},
        )
    )


def test_empty_store_search_returns_empty_results():
    assert search(InMemoryVectorStore(), [1.0, 0.0]) == []


def test_add_and_search_uses_cosine_similarity():
    store = InMemoryVectorStore()
    run(store.add([entry("CH-A", [1.0, 0.0]), entry("CH-B", [0.0, 1.0])]))

    results = search(store, [1.0, 0.0])

    assert results[0].chunk.chunk_id == "CH-A"
    assert results[0].score == pytest.approx(1.0)


def test_cosine_results_are_ranked_from_highest_to_lowest():
    store = InMemoryVectorStore()
    run(
        store.add(
            [
                entry("CH-A", [1.0, 0.0]),
                entry("CH-B", [0.8, 0.2]),
                entry("CH-C", [-1.0, 0.0]),
            ]
        )
    )

    results = search(store, [1.0, 0.0])

    assert [result.chunk.chunk_id for result in results] == ["CH-A", "CH-B", "CH-C"]
    assert results[0].score > results[1].score > results[2].score
    assert results[2].score == pytest.approx(-1.0)
    assert [result.rank for result in results] == [1, 2, 3]


def test_equal_scores_use_chunk_id_tie_break():
    store = InMemoryVectorStore()
    run(store.add([entry("CH-Z", [1.0, 0.0]), entry("CH-A", [1.0, 0.0])]))

    assert [item.chunk.chunk_id for item in search(store, [1.0, 0.0])] == [
        "CH-A",
        "CH-Z",
    ]


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"department": "support"}, ["CH-A"]),
        ({"source": "billing-guide"}, ["CH-B"]),
        ({"document_id": "DOC-C"}, ["CH-C"]),
        ({"department": "missing"}, []),
    ],
)
def test_exact_match_filters(filters, expected):
    store = InMemoryVectorStore()
    run(
        store.add(
            [
                entry("CH-A", [1.0, 0.0], metadata={"department": "support"}),
                entry("CH-B", [1.0, 0.0], source="billing-guide"),
                entry("CH-C", [1.0, 0.0], document_id="DOC-C"),
            ]
        )
    )

    assert [item.chunk.chunk_id for item in search(store, [1.0, 0.0], filters=filters)] == expected


def test_none_filter_requires_metadata_key_to_exist():
    store = InMemoryVectorStore()
    run(
        store.add(
            [
                entry("CH-MISSING", [1.0, 0.0]),
                entry("CH-EXPLICIT", [1.0, 0.0], metadata={"department": None}),
            ]
        )
    )

    results = search(store, [1.0, 0.0], filters={"department": None})

    assert [item.chunk.chunk_id for item in results] == ["CH-EXPLICIT"]


def test_same_dimension_adds_are_allowed_and_query_dimension_is_checked():
    store = InMemoryVectorStore()
    run(store.add([entry("CH-A", [1.0, 0.0])]))
    run(store.add([entry("CH-B", [0.0, 1.0])]))

    with pytest.raises(VectorStoreError):
        search(store, [1.0])


def test_existing_store_rejects_different_dimension_atomically():
    store = InMemoryVectorStore()
    run(store.add([entry("CH-A", [1.0, 0.0])]))

    with pytest.raises(VectorStoreError):
        run(store.add([entry("CH-B", [1.0])]))

    assert [item.chunk.chunk_id for item in search(store, [1.0, 0.0])] == ["CH-A"]


def test_mixed_first_batch_does_not_poison_store_dimension_or_partially_add():
    store = InMemoryVectorStore()

    with pytest.raises(VectorStoreError):
        run(store.add([entry("CH-2D", [1.0, 0.0]), entry("CH-1D", [1.0])]))

    run(store.add([entry("CH-VALID", [1.0])]))
    assert [item.chunk.chunk_id for item in search(store, [1.0])] == ["CH-VALID"]


def test_duplicate_ids_inside_batch_and_against_store_are_atomic():
    store = InMemoryVectorStore()
    duplicate = entry("CH-A", [1.0, 0.0])

    with pytest.raises(VectorStoreError):
        run(store.add([duplicate, duplicate.model_copy(deep=True)]))
    assert search(store, [1.0, 0.0]) == []

    run(store.add([duplicate]))
    with pytest.raises(VectorStoreError):
        run(store.add([entry("CH-NEW", [0.0, 1.0]), duplicate]))
    assert [item.chunk.chunk_id for item in search(store, [1.0, 0.0])] == ["CH-A"]


def test_add_empty_batch_succeeds_and_top_k_is_validated():
    store = InMemoryVectorStore()
    run(store.add([]))

    with pytest.raises(VectorStoreError):
        search(store, [1.0], top_k=0)


def test_store_defensively_copies_inputs_and_search_results():
    store = InMemoryVectorStore()
    original = entry("CH-A", [1.0, 0.0], metadata={"department": "support"})
    run(store.add([original]))
    original.chunk.content = "mutated"
    original.chunk.metadata["department"] = "changed"
    original.embedding.values[0] = -1.0

    first = search(store, [1.0, 0.0])[0]
    assert first.chunk.content == "Content for CH-A"
    assert first.chunk.metadata["department"] == "support"
    assert first.score == pytest.approx(1.0)

    first.chunk.metadata["department"] = "changed again"
    assert search(store, [1.0, 0.0])[0].chunk.metadata["department"] == "support"


def test_zero_vector_scores_zero_without_error():
    store = InMemoryVectorStore()
    run(store.add([entry("CH-ZERO", [0.0, 0.0])]))

    result = search(store, [1.0, 0.0])[0]

    assert result.score == 0.0


def test_negative_score_is_not_filtered():
    store = InMemoryVectorStore()
    run(store.add([entry("CH-NEGATIVE", [-1.0, 0.0])]))

    result = search(store, [1.0, 0.0])

    assert result[0].score == pytest.approx(-1.0)
