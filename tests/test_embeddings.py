import asyncio
import math

import pytest
from pydantic import ValidationError

from app.retrieval.embeddings import EmbeddingError, EmbeddingVector
from app.retrieval.hash_embedding import HashEmbeddingProvider


def test_embedding_vector_accepts_finite_values():
    vector = EmbeddingVector(values=[-1.5, 0.0, 2.5])

    assert vector.values == [-1.5, 0.0, 2.5]


@pytest.mark.parametrize("values", [[], [math.nan], [math.inf], [-math.inf]])
def test_embedding_vector_rejects_empty_or_non_finite_values(values):
    with pytest.raises(ValidationError):
        EmbeddingVector(values=values)


@pytest.mark.parametrize("dimensions", [0, -1])
def test_hash_embedding_rejects_invalid_dimensions(dimensions: int):
    with pytest.raises(ValueError):
        HashEmbeddingProvider(dimensions=dimensions)


def test_hash_embedding_has_requested_dimensions_and_is_deterministic():
    provider = HashEmbeddingProvider(dimensions=17)

    first = asyncio.run(provider.embed_query("login authentication"))
    second = asyncio.run(provider.embed_query("login authentication"))
    different = asyncio.run(provider.embed_query("billing invoice"))

    assert len(first.values) == 17
    assert first == second
    assert first != different
    assert all(-1 <= value <= 1 for value in first.values)


def test_embed_documents_preserves_input_order():
    provider = HashEmbeddingProvider(dimensions=8)
    texts = ["A", "B", "C"]

    batch = asyncio.run(provider.embed_documents(texts))
    individual = [asyncio.run(provider.embed_query(text)) for text in texts]

    assert batch == individual


def test_embed_documents_accepts_empty_batch():
    result = asyncio.run(HashEmbeddingProvider().embed_documents([]))

    assert result == []


@pytest.mark.parametrize("text", ["", "   "])
def test_embed_query_rejects_blank_text(text: str):
    with pytest.raises(EmbeddingError):
        asyncio.run(HashEmbeddingProvider().embed_query(text))


def test_embed_documents_rejects_entire_batch_containing_blank_text():
    provider = HashEmbeddingProvider()

    with pytest.raises(EmbeddingError):
        asyncio.run(provider.embed_documents(["valid", "   ", "also valid"]))
