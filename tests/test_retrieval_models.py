import pytest
from pydantic import ValidationError

from app.retrieval.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalQuery,
    RetrievalResult,
)


def test_knowledge_document_accepts_valid_data():
    document = KnowledgeDocument(
        document_id="DOC-001",
        title="Support handbook",
        source="support-handbook",
        content="Enterprise login support guidance.",
        metadata={"department": "support", "version": 1},
    )

    assert document.document_id == "DOC-001"
    assert document.metadata == {"department": "support", "version": 1}


@pytest.mark.parametrize("field", ["document_id", "title", "source", "content"])
def test_knowledge_document_rejects_blank_required_strings(field: str):
    values = {
        "document_id": "DOC-001",
        "title": "Support handbook",
        "source": "support-handbook",
        "content": "Support guidance",
    }
    values[field] = "   "

    with pytest.raises(ValidationError):
        KnowledgeDocument(**values)


def test_knowledge_document_rejects_non_json_metadata():
    with pytest.raises(ValidationError):
        KnowledgeDocument(
            document_id="DOC-001",
            title="Support handbook",
            source="support-handbook",
            content="Support guidance",
            metadata={"invalid": object()},
        )


def test_knowledge_chunk_accepts_valid_data():
    chunk = KnowledgeChunk(
        chunk_id="CH-001",
        document_id="DOC-001",
        content="Login troubleshooting",
        source="support-handbook",
        position=0,
    )

    assert chunk.position == 0
    assert chunk.metadata == {}


@pytest.mark.parametrize("field", ["chunk_id", "document_id", "content", "source"])
def test_knowledge_chunk_rejects_blank_required_strings(field: str):
    values = {
        "chunk_id": "CH-001",
        "document_id": "DOC-001",
        "content": "Login troubleshooting",
        "source": "support-handbook",
    }
    values[field] = " "

    with pytest.raises(ValidationError):
        KnowledgeChunk(**values)


def test_knowledge_chunk_rejects_negative_position():
    with pytest.raises(ValidationError):
        KnowledgeChunk(
            chunk_id="CH-001",
            document_id="DOC-001",
            content="Login troubleshooting",
            source="support-handbook",
            position=-1,
        )


def test_retrieval_query_has_valid_defaults():
    query = RetrievalQuery(query="login authentication")

    assert query.top_k == 5
    assert query.filters == {}


@pytest.mark.parametrize(
    "values",
    [
        {"query": "   "},
        {"query": "login", "top_k": 0},
        {"query": "login", "top_k": 21},
        {"query": "login", "filters": {"invalid": object()}},
    ],
)
def test_retrieval_query_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        RetrievalQuery(**values)


def test_retrieval_result_accepts_valid_score_and_rank():
    chunk = KnowledgeChunk(
        chunk_id="CH-001",
        document_id="DOC-001",
        content="Login troubleshooting",
        source="support-handbook",
    )

    result = RetrievalResult(chunk=chunk, score=0.75, rank=1)

    assert result.score == 0.75
    assert result.rank == 1


@pytest.mark.parametrize("score", [-0.5, 1.5])
def test_retrieval_result_score_is_backend_agnostic(score: float):
    chunk = KnowledgeChunk(
        chunk_id="CH-001",
        document_id="DOC-001",
        content="Login troubleshooting",
        source="support-handbook",
    )

    result = RetrievalResult(chunk=chunk, score=score)

    assert result.score == score


def test_retrieval_result_rejects_rank_zero():
    chunk = KnowledgeChunk(
        chunk_id="CH-001",
        document_id="DOC-001",
        content="Login troubleshooting",
        source="support-handbook",
    )

    with pytest.raises(ValidationError):
        RetrievalResult(chunk=chunk, score=0.5, rank=0)
