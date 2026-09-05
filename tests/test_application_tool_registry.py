import asyncio
from types import SimpleNamespace

import pytest

import app.api.agent as agent_api
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


def test_approval_service_uses_same_registry_for_planning_and_execution(
    monkeypatch,
):
    registry = get_tool_registry(FakeRetriever())
    captured = {}

    class FakePlannerService:
        def __init__(self, llm_service, tool_definitions=None):
            captured["planner_llm"] = llm_service
            captured["tool_definitions"] = tool_definitions

    class FakeGroundedAnswerService:
        def __init__(self, llm_service):
            captured["grounding_llm"] = llm_service

    class FakeApprovalWorkflowService:
        def __init__(
            self,
            planner_service,
            registry,
            checkpointer,
            grounded_answer_service=None,
        ):
            captured["planner_service"] = planner_service
            captured["registry"] = registry
            captured["grounded_answer_service"] = grounded_answer_service

    monkeypatch.setattr(agent_api, "PlannerService", FakePlannerService)
    monkeypatch.setattr(
        agent_api,
        "ApprovalWorkflowService",
        FakeApprovalWorkflowService,
    )
    monkeypatch.setattr(
        agent_api,
        "GroundedAnswerService",
        FakeGroundedAnswerService,
    )

    llm_service = object()
    agent_api.get_approval_workflow_service(
        llm_service=llm_service,
        checkpointer=object(),
        registry=registry,
    )

    assert captured["registry"] is registry
    assert captured["tool_definitions"] == registry.definitions()
    assert captured["planner_llm"] is llm_service
    assert captured["grounding_llm"] is llm_service
