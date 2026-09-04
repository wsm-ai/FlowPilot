import asyncio
from typing import Any

import pytest

from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse, LLMToolCall
from app.services.llm_service import LLMService
from app.services.tool_calling_service import ToolCallingService
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


def create_service(*responses: LLMResponse):
    provider = FakeProvider(list(responses))
    service = ToolCallingService(
        LLMService(provider),
        create_default_tool_registry(),
    )
    return service, provider


def feedback_call(arguments: str, call_id: str = "call_123") -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name="get_customer_feedback",
        arguments=arguments,
    )


def test_direct_answer_does_not_execute_tools():
    service, provider = create_service(LLMResponse(content="Direct answer"))

    result = asyncio.run(service.run("Hello"))

    assert result.answer == "Direct answer"
    assert result.executed_tools == []
    assert len(provider.requests) == 1


def test_executes_tool_and_returns_final_answer_with_complete_history():
    service, provider = create_service(
        LLMResponse(
            content="I will check.",
            tool_calls=[
                feedback_call('{"customer_id":"C001","priority":"high"}')
            ],
        ),
        LLMResponse(content="C001 has two high-priority feedback records."),
    )

    result = asyncio.run(service.run("Find high-priority feedback for C001"))

    assert result.answer == "C001 has two high-priority feedback records."
    assert len(result.executed_tools) == 1
    assert result.executed_tools[0].name == "get_customer_feedback"
    assert result.executed_tools[0].arguments == {
        "customer_id": "C001",
        "priority": "high",
    }

    assert len(provider.requests) == 2
    second_request = provider.requests[1]
    assert second_request["tool_choice"] == "none"
    assert second_request["tools"] == provider.requests[0]["tools"]
    history = second_request["messages"]
    assert history[0] == {
        "role": "user",
        "content": "Find high-priority feedback for C001",
    }
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "I will check."
    assert history[1]["tool_calls"][0]["id"] == "call_123"
    assert history[2]["role"] == "tool"
    assert history[2]["tool_call_id"] == "call_123"
    assert "FB-001" in history[2]["content"]
    assert "FB-003" in history[2]["content"]


@pytest.mark.parametrize(
    "arguments",
    ["{invalid json", "[1, 2]"],
)
def test_rejects_invalid_tool_argument_json(arguments):
    service, _ = create_service(
        LLMResponse(content=None, tool_calls=[feedback_call(arguments)])
    )

    with pytest.raises(ToolExecutionError):
        asyncio.run(service.run("Find feedback"))


def test_rejects_unknown_tool():
    service, _ = create_service(
        LLMResponse(
            content=None,
            tool_calls=[
                LLMToolCall(id="call_123", name="unknown_tool", arguments="{}")
            ],
        )
    )

    with pytest.raises(ToolExecutionError):
        asyncio.run(service.run("Do something"))


def test_rejects_invalid_tool_parameters():
    service, _ = create_service(
        LLMResponse(
            content=None,
            tool_calls=[
                feedback_call('{"customer_id":"C001","priority":"urgent"}')
            ],
        )
    )

    with pytest.raises(ToolExecutionError):
        asyncio.run(service.run("Find feedback"))


def test_executes_multiple_tool_calls_before_one_final_request():
    service, provider = create_service(
        LLMResponse(
            content=None,
            tool_calls=[
                feedback_call('{"customer_id":"C001"}', "call_1"),
                feedback_call('{"customer_id":"C002"}', "call_2"),
            ],
        ),
        LLMResponse(content="Combined answer"),
    )

    result = asyncio.run(service.run("Find both customers"))

    assert [item.tool_call_id for item in result.executed_tools] == [
        "call_1",
        "call_2",
    ]
    assert len(provider.requests) == 2
    tool_messages = [
        item
        for item in provider.requests[1]["messages"]
        if item["role"] == "tool"
    ]
    assert [item["tool_call_id"] for item in tool_messages] == ["call_1", "call_2"]


def test_requires_content_from_second_response():
    service, _ = create_service(
        LLMResponse(
            content=None,
            tool_calls=[feedback_call('{"customer_id":"C001"}')],
        ),
        LLMResponse(content=None),
    )

    with pytest.raises(LLMProviderError):
        asyncio.run(service.run("Find feedback"))
