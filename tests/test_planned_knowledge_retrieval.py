import asyncio
import json

import pytest

from app.providers.types import LLMResponse
from app.retrieval.chunking import SimpleTextChunker
from app.retrieval.demo_knowledge import DEMO_KNOWLEDGE_DOCUMENTS, bootstrap_demo_knowledge_base
from app.retrieval.hash_embedding import HashEmbeddingProvider
from app.retrieval.in_memory_vector_store import InMemoryVectorStore
from app.retrieval.indexing import KnowledgeBaseIndexer
from app.retrieval.ingestion import DocumentIngestionService
from app.retrieval.models import KnowledgeChunk, RetrievalQuery, RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever
from app.services.llm_service import LLMService
from app.services.planned_agent_service import PlannedAgentService
from app.services.planner_service import PlannerService
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import create_default_tool_registry


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[RetrievalQuery] = []

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        return [
            RetrievalResult(
                chunk=KnowledgeChunk(
                    chunk_id="KB-1:chunk:0",
                    document_id="KB-1",
                    content="Enterprise login guidance",
                    source="internal-guide",
                    metadata={"department": "support"},
                    position=0,
                ),
                score=0.9,
                rank=1,
            )
        ]


class PlanProvider:
    model = "test-model"

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(
                {
                    "goal": "Find internal login guidance",
                    "steps": [
                        {
                            "id": 1,
                            "description": "Search internal knowledge",
                            "action": "search_knowledge_base",
                            "arguments": {"query": "Enterprise login troubleshooting"},
                            "requires_approval": True,
                        }
                    ],
                }
            )
        )


def application_registry(retriever):
    registry = create_default_tool_registry()
    registry.register(KnowledgeBaseTool(retriever))
    return registry


def test_planner_catalog_and_execution_registry_capture_retrieval_evidence():
    """Static Stage 8D plan captures evidence; later synthesis belongs to Stage 8E."""
    retriever = FakeRetriever()
    registry = application_registry(retriever)
    planner = PlannerService(
        LLMService(PlanProvider()), tool_definitions=registry.definitions()
    )

    result = asyncio.run(
        PlannedAgentService(planner, registry).run("Find internal login guidance")
    )

    assert result.status == "completed"
    assert result.current_step_index == 1
    assert len(result.step_results) == 1
    step_result = result.step_results[0]
    assert step_result["step_id"] == 1
    assert step_result["action"] == "search_knowledge_base"
    assert set(step_result["result"][0]) >= {
        "chunk_id", "document_id", "content", "source", "score", "rank", "metadata"
    }
    assert len(retriever.queries) == 1


def test_demo_bootstrap_vector_pipeline_executes_through_registry():
    async def scenario():
        embeddings = HashEmbeddingProvider()
        store = InMemoryVectorStore()
        ingestion = DocumentIngestionService(SimpleTextChunker(), embeddings)
        await bootstrap_demo_knowledge_base(KnowledgeBaseIndexer(ingestion, store))
        registry = application_registry(VectorRetriever(embeddings, store))
        return await registry.execute(
            "search_knowledge_base",
            {"query": DEMO_KNOWLEDGE_DOCUMENTS[0].content, "top_k": 1},
        )

    result = asyncio.run(scenario())

    assert result[0]["document_id"] == "DOC-001"
    assert result[0]["score"] == pytest.approx(1.0)
