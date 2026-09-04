import asyncio
import json
from typing import Any

import pytest

from app.graph.planning_workflow import create_planning_graph
from app.providers.types import LLMResponse
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, PlanningError


GOAL = "Analyze high-priority customer feedback and prepare missing issues."

VALID_PLAN = {
    "goal": GOAL,
    "steps": [
        {
            "id": 1,
            "description": "Retrieve high-priority customer feedback",
            "action": "get_customer_feedback",
            "requires_approval": False,
        }
    ],
}


class FakeProvider:
    model = "test-model"

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(content=json.dumps(VALID_PLAN))


def test_planning_graph_compiles_and_creates_plan_without_tools():
    provider = FakeProvider()
    planner_service = PlannerService(LLMService(provider))
    graph = create_planning_graph(planner_service)

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": [],
                "llm_response": None,
                "answer": None,
                "executed_tools": [],
                "goal": GOAL,
                "plan": None,
            }
        )
    )

    assert graph is not None
    assert "planner" in graph.get_graph().nodes
    assert provider.call_count == 1
    assert result["plan"].goal == VALID_PLAN["goal"]
    assert len(result["plan"].steps) == len(VALID_PLAN["steps"])
    assert result["executed_tools"] == []


def test_planning_graph_rejects_state_without_goal():
    provider = FakeProvider()
    graph = create_planning_graph(PlannerService(LLMService(provider)))

    with pytest.raises(PlanningError, match="Planning goal is required"):
        asyncio.run(
            graph.ainvoke(
                {
                    "messages": [],
                    "llm_response": None,
                    "answer": None,
                    "executed_tools": [],
                }
            )
        )

    assert provider.call_count == 0
