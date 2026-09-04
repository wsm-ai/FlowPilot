import asyncio
from typing import Any

import pytest

from app.graph.workflow import create_basic_agent_graph
from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse, LLMToolCall
from app.services.llm_service import LLMService
from app.tools.base import ToolExecutionError
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


def create_graph(*responses: LLMResponse):
    provider = FakeProvider(list(responses))
    registry = create_default_tool_registry()
    graph = create_basic_agent_graph(LLMService(provider), registry)
    return graph, provider


def initial_state(message: str = "What is an AI Agent?") -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": message}],
        "llm_response": None,
        "answer": None,
        "executed_tools": [],
    }


def feedback_call(arguments: str, name: str = "get_customer_feedback"):
    return LLMToolCall(
        id="call_123",
        name=name,
        arguments=arguments,
    )


def test_basic_agent_graph_compiles():
    graph, _ = create_graph(LLMResponse(content="Hello"))

    assert graph is not None
    assert {"llm", "tools", "final_llm"}.issubset(graph.get_graph().nodes)


def test_direct_answer_routes_to_end_without_executing_tools():
    expected_response = LLMResponse(content="An AI agent can act toward a goal.")
    graph, provider = create_graph(expected_response)
    state = initial_state()

    result = asyncio.run(graph.ainvoke(state))

    assert len(provider.requests) == 1
    assert provider.requests[0]["messages"] == state["messages"]
    assert provider.requests[0]["tools"]
    assert provider.requests[0]["tool_choice"] == "auto"
    assert result["answer"] == "An AI agent can act toward a goal."
    assert result["llm_response"] is expected_response
    assert result["executed_tools"] == []
    assert result["messages"] == [
        *state["messages"],
        {
            "role": "assistant",
            "content": "An AI agent can act toward a goal.",
        },
    ]


def test_tool_call_routes_to_tool_node_and_updates_state():
    selection_response = LLMResponse(
        content=None,
        tool_calls=[
            feedback_call('{"customer_id":"C001","priority":"high"}')
        ],
    )
    final_response = LLMResponse(
        content="Customer C001 has two high-priority issues."
    )
    graph, provider = create_graph(selection_response, final_response)

    result = asyncio.run(graph.ainvoke(initial_state("Find C001 high feedback")))

    assert len(provider.requests) == 2
    assert provider.requests[0]["tool_choice"] == "auto"
    assert provider.requests[1]["tool_choice"] == "none"
    assert result["answer"] == "Customer C001 has two high-priority issues."
    assert result["llm_response"] is final_response
    assert result["executed_tools"] == [
        {
            "tool_call_id": "call_123",
            "name": "get_customer_feedback",
            "arguments": {"customer_id": "C001", "priority": "high"},
        }
    ]
    second_request_messages = provider.requests[1]["messages"]
    assert second_request_messages[0]["role"] == "user"
    assistant_message = second_request_messages[1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["tool_calls"][0]["id"] == "call_123"
    tool_message = second_request_messages[2]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_123"
    assert "FB-001" in tool_message["content"]
    assert "FB-003" in tool_message["content"]
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "Customer C001 has two high-priority issues.",
    }


def test_tool_call_content_is_not_treated_as_final_answer():
    selection_response = LLMResponse(
        content="I'll query the customer feedback.",
        tool_calls=[
            feedback_call('{"customer_id":"C001","priority":"high"}')
        ],
    )
    graph, provider = create_graph(
        selection_response,
        LLMResponse(content="Final answer"),
    )

    result = asyncio.run(graph.ainvoke(initial_state("Find C001 high feedback")))

    assert result["answer"] == "Final answer"
    assert result["executed_tools"][0]["name"] == "get_customer_feedback"
    assert provider.requests[1]["messages"][1]["content"] == (
        "I'll query the customer feedback."
    )
    tool_message = provider.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "FB-001" in tool_message["content"]
    assert "FB-003" in tool_message["content"]


@pytest.mark.parametrize("arguments", ["{invalid", "[1,2]"])
def test_tool_node_rejects_invalid_arguments(arguments: str):
    graph, _ = create_graph(
        LLMResponse(content=None, tool_calls=[feedback_call(arguments)])
    )

    with pytest.raises(ToolExecutionError):
        asyncio.run(graph.ainvoke(initial_state("Find feedback")))


def test_tool_node_rejects_unknown_tool():
    graph, _ = create_graph(
        LLMResponse(content=None, tool_calls=[feedback_call("{}", "unknown_tool")])
    )

    with pytest.raises(ToolExecutionError):
        asyncio.run(graph.ainvoke(initial_state("Use unknown tool")))


@pytest.mark.parametrize("content", [None, ""])
def test_final_llm_requires_content(content: str | None):
    graph, provider = create_graph(
        LLMResponse(
            content=None,
            tool_calls=[
                feedback_call('{"customer_id":"C001","priority":"high"}')
            ],
        ),
        LLMResponse(content=content),
    )

    with pytest.raises(LLMProviderError):
        asyncio.run(graph.ainvoke(initial_state("Find feedback")))

    assert len(provider.requests) == 2
    assert provider.requests[1]["tool_choice"] == "none"
