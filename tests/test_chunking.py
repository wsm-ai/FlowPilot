import pytest

from app.retrieval.chunking import SimpleTextChunker
from app.retrieval.models import KnowledgeDocument


def document(content: str = "Short text") -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id="DOC-001",
        title="Support Handbook",
        source="support-handbook",
        content=content,
        metadata={"department": "support"},
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"chunk_size": 0},
        {"chunk_size": -1},
        {"chunk_size": 10, "chunk_overlap": -1},
        {"chunk_size": 10, "chunk_overlap": 10},
        {"chunk_size": 10, "chunk_overlap": 11},
    ],
)
def test_chunker_rejects_invalid_configuration(arguments):
    with pytest.raises(ValueError):
        SimpleTextChunker(**arguments)


def test_short_document_produces_one_chunk():
    chunks = SimpleTextChunker(chunk_size=20, chunk_overlap=2).chunk(document())

    assert len(chunks) == 1
    assert chunks[0].content == "Short text"
    assert chunks[0].position == 0


def test_document_equal_to_chunk_size_produces_exactly_one_chunk():
    chunks = SimpleTextChunker(chunk_size=10, chunk_overlap=2).chunk(
        document("ABCDEFGHIJ")
    )

    assert [chunk.content for chunk in chunks] == ["ABCDEFGHIJ"]


def test_chunker_stops_when_previous_window_reaches_document_end():
    chunks = SimpleTextChunker(chunk_size=6, chunk_overlap=2).chunk(
        document("ABCDEFGHIJ")
    )

    assert [chunk.content for chunk in chunks] == ["ABCDEF", "EFGHIJ"]


def test_long_document_uses_deterministic_overlapping_windows():
    chunker = SimpleTextChunker(chunk_size=6, chunk_overlap=2)

    first = chunker.chunk(document("ABCDEFGHIJKLMNO"))
    second = chunker.chunk(document("ABCDEFGHIJKLMNO"))

    assert [chunk.content for chunk in first] == [
        "ABCDEF",
        "EFGHIJ",
        "IJKLMN",
        "MNO",
    ]
    assert [chunk.position for chunk in first] == [0, 1, 2, 3]
    assert [chunk.chunk_id for chunk in first] == [
        "DOC-001:chunk:0",
        "DOC-001:chunk:1",
        "DOC-001:chunk:2",
        "DOC-001:chunk:3",
    ]
    assert [chunk.chunk_id for chunk in second] == [
        chunk.chunk_id for chunk in first
    ]


def test_chunk_metadata_is_propagated_without_sharing_document_state():
    source_document = document()

    chunk = SimpleTextChunker().chunk(source_document)[0]

    assert chunk.metadata == {
        "department": "support",
        "document_title": "Support Handbook",
    }
    chunk.metadata["department"] = "changed"
    assert source_document.metadata == {"department": "support"}
