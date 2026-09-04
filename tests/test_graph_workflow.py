import asyncio
from typing import Any

from app.graph.workflow import create_basic_agent_graph
from app.providers.types import LLMResponse
from app.services.llm_service import LLMService


class FakeProvider:
    model = "test-model"

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.requests.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return self.response


def test_basic_agent_graph_compiles():
    provider = FakeProvider(LLMResponse(content="Hello"))

    graph = create_basic_agent_graph(LLMService(provider))

    assert graph is not None
    assert "llm" in graph.get_graph().nodes


def test_basic_agent_graph_runs_llm_node_and_updates_state():
    expected_response = LLMResponse(content="An AI agent can act toward a goal.")
    provider = FakeProvider(expected_response)
    graph = create_basic_agent_graph(LLMService(provider))
    messages = [
        {
            "role": "user",
            "content": "What is an AI Agent?",
        }
    ]

    result = asyncio.run(
        graph.ainvoke(
            {
                "messages": messages,
                "llm_response": None,
                "answer": None,
            }
        )
    )

    assert len(provider.requests) == 1
    assert provider.requests[0] == {
        "messages": messages,
        "tools": None,
        "tool_choice": None,
    }
    assert result["answer"] == "An AI agent can act toward a goal."
    assert result["llm_response"] is expected_response
    assert result["messages"] == messages
