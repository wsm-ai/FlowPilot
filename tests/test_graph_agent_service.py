import asyncio
from typing import Any

from app.providers.types import LLMResponse, LLMToolCall
from app.services.graph_agent_service import GraphAgentService
from app.services.llm_service import LLMService
from app.tools.registry import create_default_tool_registry


class FakeProvider:
    model = "test-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return next(self._responses)


def create_service(*responses: LLMResponse):
    provider = FakeProvider(list(responses))
    service = GraphAgentService(
        llm_service=LLMService(provider),
        registry=create_default_tool_registry(),
    )
    return service, provider


def test_direct_answer_calls_llm_once():
    service, provider = create_service(LLMResponse(content="Direct answer"))

    result = asyncio.run(service.run("Hello"))

    assert len(provider.requests) == 1
    assert result.answer == "Direct answer"
    assert result.executed_tools == []


def test_tool_calling_calls_llm_twice_and_returns_graph_result():
    service, provider = create_service(
        LLMResponse(
            content=None,
            tool_calls=[
                LLMToolCall(
                    id="call_123",
                    name="get_customer_feedback",
                    arguments='{"customer_id":"C001","priority":"high"}',
                )
            ],
        ),
        LLMResponse(content="Customer C001 has two high-priority issues."),
    )

    result = asyncio.run(service.run("Find high-priority feedback for C001"))

    assert len(provider.requests) == 2
    assert provider.requests[0]["tool_choice"] == "auto"
    assert provider.requests[1]["tool_choice"] == "none"
    assert result.answer == "Customer C001 has two high-priority issues."
    assert len(result.executed_tools) == 1
    assert result.executed_tools[0].tool_call_id == "call_123"
    assert result.executed_tools[0].name == "get_customer_feedback"
    assert result.executed_tools[0].arguments == {
        "customer_id": "C001",
        "priority": "high",
    }
