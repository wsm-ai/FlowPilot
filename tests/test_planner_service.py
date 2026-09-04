import asyncio
import json
from typing import Any

import pytest

from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, PlanningError


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


def create_service(content: str | None):
    provider = FakeProvider(content)
    return PlannerService(LLMService(provider)), provider


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
