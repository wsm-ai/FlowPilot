import asyncio
from types import SimpleNamespace

import pytest

import app.api.agent as agent_api
from app.api.dependencies import (
    get_reactive_tool_registry,
    get_retriever,
    get_tool_registry,
)
from app.retrieval.models import KnowledgeChunk, RetrievalResult
from app.tools.knowledge_base import KnowledgeBaseTool
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


class MarkerTool:
    description = "Marker"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, name):
        self.name = name

    async def execute(self, arguments):
        return None


def request_with_state(**state_values):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(**state_values))
    )


def registry_with_knowledge(retriever):
    registry = create_default_tool_registry()
    registry.register(KnowledgeBaseTool(retriever))
    return registry


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
    registry = registry_with_knowledge(FakeRetriever())
    registry = get_tool_registry(request_with_state(tool_registry=registry))

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


def test_reactive_registry_excludes_approval_required_actions():
    registry = registry_with_knowledge(FakeRetriever())
    registry.register(MarkerTool("mcp_test_create_issue"))
    request = request_with_state(
        tool_registry=registry,
        mcp_approval_required_actions=frozenset({"mcp_test_create_issue"}),
    )

    restricted = get_reactive_tool_registry(request)
    names = {
        definition["function"]["name"]
        for definition in restricted.definitions()
    }

    assert names == {"get_customer_feedback", "search_knowledge_base"}
    assert registry.contains("mcp_test_create_issue")


def test_empty_approval_policy_keeps_all_reactive_tools():
    registry = registry_with_knowledge(FakeRetriever())
    request = request_with_state(
        tool_registry=registry,
        mcp_approval_required_actions=frozenset(),
    )
    assert get_reactive_tool_registry(request).definitions() == (
        registry.definitions()
    )


def test_knowledge_base_tool_uses_injected_shared_retriever():
    retriever = FakeRetriever()
    registry = registry_with_knowledge(retriever)
    registry = get_tool_registry(request_with_state(tool_registry=registry))

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
    registry = registry_with_knowledge(FakeRetriever())
    captured = {}

    class FakePlannerService:
        def __init__(
            self,
            llm_service,
            tool_definitions=None,
            approval_required_actions=None,
        ):
            captured["planner_llm"] = llm_service
            captured["tool_definitions"] = tool_definitions
            captured["approval_required_actions"] = approval_required_actions

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
            side_effect_executor=None,
        ):
            captured["planner_service"] = planner_service
            captured["registry"] = registry
            captured["grounded_answer_service"] = grounded_answer_service
            captured["side_effect_executor"] = side_effect_executor

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
    side_effect_executor = object()
    agent_api.get_approval_workflow_service(
        llm_service=llm_service,
        checkpointer=object(),
        registry=registry,
        approval_required_actions=frozenset({"mcp_test_action"}),
        side_effect_executor=side_effect_executor,
    )

    assert captured["registry"] is registry
    assert captured["tool_definitions"] == registry.definitions()
    assert captured["planner_llm"] is llm_service
    assert captured["grounding_llm"] is llm_service
    assert captured["approval_required_actions"] == {"mcp_test_action"}
    assert captured["side_effect_executor"] is side_effect_executor
