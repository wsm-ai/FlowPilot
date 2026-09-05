import asyncio
import json

import pytest

from app.retrieval.base import RetrievalError
from app.retrieval.embeddings import EmbeddingError
from app.retrieval.models import KnowledgeChunk, RetrievalResult
from app.retrieval.vector_store import VectorStoreError
from app.tools.base import ToolExecutionError
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import ToolRegistry, create_default_tool_registry


class FakeRetriever:
    def __init__(self, results=None, error=None) -> None:
        self.results = [] if results is None else results
        self.error = error
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.results


def retrieval_result(
    *,
    score: float = 0.91,
    chunk_id: str = "CH-001",
    content: str = "Enterprise login troubleshooting",
) -> RetrievalResult:
    return RetrievalResult(
        chunk=KnowledgeChunk(
            chunk_id=chunk_id,
            document_id="DOC-001",
            content=content,
            source="support-handbook",
            metadata={
                "department": "support",
                "document_title": "Support Handbook",
            },
            position=0,
        ),
        score=score,
        rank=1,
    )


def test_tool_metadata_and_generated_parameters_schema():
    tool = KnowledgeBaseTool(FakeRetriever())

    assert tool.name == "search_knowledge_base"
    assert tool.description
    schema = tool.parameters
    assert schema["type"] == "object"
    assert {"query", "top_k", "filters"}.issubset(schema["properties"])
    assert schema["properties"]["top_k"]["minimum"] == 1
    assert schema["properties"]["top_k"]["maximum"] == 20


def test_valid_execution_forwards_retrieval_query_and_returns_evidence():
    retriever = FakeRetriever([retrieval_result()])
    tool = KnowledgeBaseTool(retriever)

    output = asyncio.run(
        tool.execute(
            {
                "query": "enterprise login",
                "top_k": 3,
                "filters": {"department": "support"},
            }
        )
    )

    received = retriever.queries[0]
    assert received.query == "enterprise login"
    assert received.top_k == 3
    assert received.filters == {"department": "support"}
    assert output == [
        {
            "chunk_id": "CH-001",
            "document_id": "DOC-001",
            "content": "Enterprise login troubleshooting",
            "source": "support-handbook",
            "score": 0.91,
            "rank": 1,
            "position": 0,
            "metadata": {
                "department": "support",
                "document_title": "Support Handbook",
            },
        }
    ]
    json.dumps(output, allow_nan=False)


def test_default_arguments_are_forwarded():
    retriever = FakeRetriever()

    result = asyncio.run(KnowledgeBaseTool(retriever).execute({"query": "login"}))

    assert result == []
    assert retriever.queries[0].top_k == 5
    assert retriever.queries[0].filters == {}


def test_zero_results_are_not_an_error():
    result = asyncio.run(
        KnowledgeBaseTool(FakeRetriever()).execute({"query": "not found"})
    )

    assert result == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "login", "top_k": 0},
        {"query": "login", "top_k": 21},
        {"query": "login", "filters": {"bad": object()}},
        {"query": "login", "unknown_field": "x"},
    ],
)
def test_invalid_arguments_raise_safe_tool_error(arguments):
    with pytest.raises(ToolExecutionError) as raised:
        asyncio.run(KnowledgeBaseTool(FakeRetriever()).execute(arguments))

    assert str(raised.value) == "Invalid knowledge base search arguments"


@pytest.mark.parametrize(
    "error",
    [
        RetrievalError("sensitive retrieval detail"),
        EmbeddingError("secret embedding detail"),
        VectorStoreError("private db detail"),
    ],
)
def test_infrastructure_errors_are_mapped_without_leaking_details(error):
    with pytest.raises(ToolExecutionError) as raised:
        asyncio.run(
            KnowledgeBaseTool(FakeRetriever(error=error)).execute(
                {"query": "login"}
            )
        )

    assert str(raised.value) == "Knowledge base search failed"
    assert raised.value.__cause__ is error
    assert str(error) not in str(raised.value)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_non_json_score_is_rejected_at_tool_boundary(score: float):
    tool = KnowledgeBaseTool(FakeRetriever([retrieval_result(score=score)]))

    with pytest.raises(ToolExecutionError) as raised:
        asyncio.run(tool.execute({"query": "login"}))

    assert str(raised.value) == "Knowledge base search returned invalid data"


def test_output_metadata_is_a_defensive_copy():
    original = retrieval_result()
    output = asyncio.run(
        KnowledgeBaseTool(FakeRetriever([original])).execute({"query": "login"})
    )

    output[0]["metadata"]["department"] = "changed"

    assert original.chunk.metadata["department"] == "support"


def test_unexpected_error_is_left_for_registry_to_wrap():
    unexpected = RuntimeError("private unexpected detail")
    tool = KnowledgeBaseTool(FakeRetriever(error=unexpected))

    with pytest.raises(RuntimeError) as direct:
        asyncio.run(tool.execute({"query": "login"}))
    assert direct.value is unexpected

    registry = ToolRegistry()
    registry.register(tool)
    with pytest.raises(ToolExecutionError) as wrapped:
        asyncio.run(registry.execute("search_knowledge_base", {"query": "login"}))
    assert str(wrapped.value) == "Tool execution failed: search_knowledge_base"
    assert wrapped.value.__cause__ is unexpected


def test_tool_registry_can_register_define_and_execute_knowledge_base_tool():
    retriever = FakeRetriever([retrieval_result()])
    registry = ToolRegistry()
    registry.register(KnowledgeBaseTool(retriever))

    definitions = registry.definitions()
    result = asyncio.run(
        registry.execute("search_knowledge_base", {"query": "login"})
    )

    assert definitions[0]["function"]["name"] == "search_knowledge_base"
    assert result[0]["chunk_id"] == "CH-001"
    assert len(retriever.queries) == 1


def test_default_registry_remains_unchanged():
    definitions = create_default_tool_registry().definitions()

    assert [item["function"]["name"] for item in definitions] == [
        "get_customer_feedback"
    ]


def test_tool_defensively_enforces_requested_top_k():
    results = [retrieval_result(chunk_id=f"CH-{index}") for index in range(5)]
    output = asyncio.run(
        KnowledgeBaseTool(FakeRetriever(results)).execute(
            {"query": "login", "top_k": 2}
        )
    )
    assert [item["chunk_id"] for item in output] == ["CH-0", "CH-1"]


def test_oversized_serialized_tool_output_is_rejected():
    tool = KnowledgeBaseTool(
        FakeRetriever([retrieval_result(content="x" * 500)]),
        max_output_chars=100,
    )
    with pytest.raises(
        ToolExecutionError,
        match="Knowledge base search result is too large",
    ):
        asyncio.run(tool.execute({"query": "login"}))


def test_output_limit_configuration_must_be_positive():
    with pytest.raises(ValueError, match="max_output_chars must be at least 1"):
        KnowledgeBaseTool(FakeRetriever(), max_output_chars=0)
