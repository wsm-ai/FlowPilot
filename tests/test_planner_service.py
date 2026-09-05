import asyncio
import json
from typing import Any

import pytest

from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse
from app.graph.routing import route_plan_step
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.customer_feedback import CustomerFeedbackTool
from app.tools.knowledge_base import KnowledgeBaseTool
from app.tools.registry import ToolRegistry


GOAL = (
    "Analyze high-priority customer feedback, check related GitHub issues, "
    "and prepare missing issues for approval."
)

VALID_PLAN = {
    "goal": "Analyze high-priority customer feedback and prepare missing GitHub issues",
    "steps": [
        {
            "id": 1,
            "description": "Retrieve high-priority customer feedback",
            "action": "get_customer_feedback",
            "arguments": {"customer_id": "C001", "priority": "high"},
            "requires_approval": False,
        },
        {
            "id": 2,
            "description": "Search related GitHub issues",
            "action": "search_github_issues",
            "requires_approval": False,
        },
        {
            "id": 3,
            "description": "Compare customer feedback with existing issues",
            "action": "compare_results",
            "requires_approval": False,
        },
        {
            "id": 4,
            "description": "Prepare missing GitHub issues for review",
            "action": "prepare_github_issue",
            "requires_approval": True,
        },
    ],
}


class FakeProvider:
    model = "test-model"

    def __init__(self, content: str | None) -> None:
        self.content = content
        self.requests: list[dict[str, Any]] = []

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return LLMResponse(content=self.content)


def create_service(content: str | None, tool_definitions=None):
    provider = FakeProvider(content)
    return PlannerService(
        LLMService(provider),
        tool_definitions=tool_definitions,
    ), provider


class FakeRetriever:
    async def search(self, query):
        return []


def application_tool_definitions():
    registry = ToolRegistry()
    registry.register(CustomerFeedbackTool())
    registry.register(KnowledgeBaseTool(FakeRetriever()))
    return registry.definitions()


def test_valid_json_is_parsed_into_execution_plan():
    service, provider = create_service(json.dumps(VALID_PLAN))

    plan = asyncio.run(service.create_plan(GOAL))

    assert plan.goal == VALID_PLAN["goal"]
    assert len(plan.steps) == 4
    assert [step.id for step in plan.steps] == [1, 2, 3, 4]
    assert plan.steps[0].description == "Retrieve high-priority customer feedback"
    assert plan.steps[0].action == "get_customer_feedback"
    assert plan.steps[0].arguments == {
        "customer_id": "C001",
        "priority": "high",
    }
    assert plan.steps[0].requires_approval is False
    assert plan.steps[3].requires_approval is True
    assert provider.requests[0]["messages"][0]["role"] == "system"
    assert provider.requests[0]["messages"][1] == {
        "role": "user",
        "content": GOAL,
    }
    assert provider.requests[0]["tools"] is None


def test_invalid_json_raises_planning_error():
    service, _ = create_service("{invalid")

    with pytest.raises(PlanningError):
        asyncio.run(service.create_plan(GOAL))


def test_invalid_schema_raises_planning_error():
    service, _ = create_service(json.dumps({"goal": "Goal", "steps": []}))

    with pytest.raises(PlanningError):
        asyncio.run(service.create_plan(GOAL))


@pytest.mark.parametrize("content", [None, ""])
def test_empty_llm_content_raises_provider_error(content: str | None):
    service, _ = create_service(content)

    with pytest.raises(LLMProviderError):
        asyncio.run(service.create_plan(GOAL))


def test_tool_catalog_and_knowledge_schema_are_added_to_prompt():
    plan_data = {
        "goal": "Search internal knowledge",
        "steps": [
            {
                "id": 1,
                "description": "Search internal knowledge",
                "action": "search_knowledge_base",
                "arguments": {"query": "enterprise login"},
                "requires_approval": False,
            }
        ],
    }
    service, provider = create_service(
        json.dumps(plan_data),
        application_tool_definitions(),
    )

    asyncio.run(service.create_plan("Search internal knowledge"))

    prompt = provider.requests[0]["messages"][0]["content"]
    assert "AVAILABLE TOOLS" in prompt
    assert "get_customer_feedback" in prompt
    assert "search_knowledge_base" in prompt
    assert '"query"' in prompt
    assert '"top_k"' in prompt
    assert '"filters"' in prompt


def test_knowledge_search_approval_is_forced_off_by_code_policy():
    plan_data = {
        "goal": "Search internal knowledge",
        "steps": [
            {
                "id": 1,
                "description": "Search internal knowledge",
                "action": "search_knowledge_base",
                "arguments": {"query": "enterprise login", "top_k": 3},
                "requires_approval": True,
            }
        ],
    }
    service, _ = create_service(
        json.dumps(plan_data),
        application_tool_definitions(),
    )

    plan = asyncio.run(service.create_plan("Search internal knowledge"))

    assert plan.steps[0].requires_approval is False
    assert route_plan_step({"plan": plan, "current_step_index": 0}) == "execute"


def test_unavailable_action_is_rejected_before_execution():
    definitions = [application_tool_definitions()[1]]
    plan_data = {
        "goal": "Invent action",
        "steps": [
            {
                "id": 1,
                "description": "Invent action",
                "action": "invented_tool",
            }
        ],
    }
    service, _ = create_service(json.dumps(plan_data), definitions)

    with pytest.raises(PlanningError, match="Plan contains unavailable action"):
        asyncio.run(service.create_plan("Invent action"))


def test_mixed_read_only_plan_is_allowed_and_approval_is_forced_off():
    plan_data = {
        "goal": "Review feedback and internal evidence",
        "steps": [
            {
                "id": 1,
                "description": "Read customer feedback",
                "action": "get_customer_feedback",
                "arguments": {"customer_id": "C001"},
                "requires_approval": True,
            },
            {
                "id": 2,
                "description": "Search internal evidence",
                "action": "search_knowledge_base",
                "arguments": {"query": "enterprise login"},
                "requires_approval": True,
            },
        ],
    }
    service, _ = create_service(
        json.dumps(plan_data),
        application_tool_definitions(),
    )

    plan = asyncio.run(service.create_plan(plan_data["goal"]))

    assert [step.action for step in plan.steps] == [
        "get_customer_feedback",
        "search_knowledge_base",
    ]
    assert [step.requires_approval for step in plan.steps] == [False, False]


def test_explicit_empty_tool_catalog_rejects_non_empty_plan():
    plan_data = {
        "goal": "Run action",
        "steps": [{"id": 1, "description": "Run", "action": "some_tool"}],
    }
    service, _ = create_service(json.dumps(plan_data), [])

    with pytest.raises(PlanningError, match="Plan contains unavailable action"):
        asyncio.run(service.create_plan("Run action"))


@pytest.mark.parametrize(
    "definition",
    [
        {"type": "function", "function": {"description": "Missing name", "parameters": {}}},
        {"type": "function", "function": {"name": "bad", "description": "Bad", "parameters": object()}},
        {"type": "invalid", "function": {"name": "bad", "description": "Bad", "parameters": {}}},
    ],
)
def test_malformed_or_non_json_tool_definition_is_rejected(definition):
    with pytest.raises(PlanningError, match="Invalid tool definition"):
        create_service(json.dumps(VALID_PLAN), [definition])
