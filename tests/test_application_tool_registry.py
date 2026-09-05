import asyncio
from types import SimpleNamespace

import pytest

from app.api.dependencies import get_retriever, get_tool_registry
from app.retrieval.models import KnowledgeChunk, RetrievalResult
from app.tools.registry import create_default_tool_registry


class FakeRetriever:
    def __init__(self) -> None:
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return [
            RetrievalResult(
                chunk=KnowledgeChunk(
                    chunk_id="CH-001",
                    document_id="DOC-001",
                    content="Enterprise login troubleshooting",
                    source="support-handbook",
                    metadata={"department": "support"},
                    position=0,
                ),
                score=1.0,
                rank=1,
            )
        ]


def request_with_state(**state_values):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**state_values))
    )


def test_get_retriever_returns_shared_app_state_instance():
    retriever = FakeRetriever()

    resolved = get_retriever(request_with_state(retriever=retriever))

    assert resolved is retriever


def test_get_retriever_fails_when_lifespan_state_is_missing():
    with pytest.raises(
        RuntimeError,
        match="Knowledge base retriever is not initialized",
    ):
        get_retriever(request_with_state())


def test_application_registry_contains_base_and_knowledge_tools():
    registry = get_tool_registry(FakeRetriever())

    names = {
        definition["function"]["name"]
        for definition in registry.definitions()
    }

    assert names == {"get_customer_feedback", "search_knowledge_base"}


def test_default_registry_remains_base_tools_only():
    names = {
        definition["function"]["name"]
        for definition in create_default_tool_registry().definitions()
    }

    assert names == {"get_customer_feedback"}


def test_knowledge_base_tool_uses_injected_shared_retriever():
    retriever = FakeRetriever()
    registry = get_tool_registry(retriever)

    result = asyncio.run(
        registry.execute(
            "search_knowledge_base",
            {"query": "enterprise login"},
        )
    )

    assert len(retriever.queries) == 1
    assert retriever.queries[0].query == "enterprise login"
    assert result[0]["chunk_id"] == "CH-001"
