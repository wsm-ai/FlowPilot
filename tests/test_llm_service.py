import asyncio

import pytest

from app.providers.base import LLMProviderError
from app.providers.types import LLMResponse, LLMToolCall
from app.services.llm_service import LLMService


class FakeProvider:
    model = "test-model"

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.last_tools = None
        self.last_tool_choice = None

    async def complete(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        return self.response


def test_chat_returns_normal_content():
    service = LLMService(FakeProvider(LLMResponse(content="Hello")))

    result = asyncio.run(service.chat("Hi"))

    assert result == "Hello"


def test_choose_tools_returns_structured_tool_call():
    tool_call = LLMToolCall(
        id="call_123",
        name="get_customer_feedback",
        arguments='{"customer_id":"C001"}',
    )
    provider = FakeProvider(LLMResponse(content=None, tool_calls=[tool_call]))
    service = LLMService(provider)
    tools = [{"type": "function", "function": {"name": "get_customer_feedback"}}]

    response = asyncio.run(service.choose_tools("Find feedback", tools))

    assert response.content is None
    assert response.tool_calls == [tool_call]
    assert response.tool_calls[0].id == "call_123"
    assert response.tool_calls[0].name == "get_customer_feedback"
    assert response.tool_calls[0].arguments == '{"customer_id":"C001"}'
    assert provider.last_tools == tools
    assert provider.last_tool_choice == "auto"


def test_chat_rejects_response_without_content():
    response = LLMResponse(
        content=None,
        tool_calls=[
            LLMToolCall(
                id="call_123",
                name="get_customer_feedback",
                arguments="{}",
            )
        ],
    )
    service = LLMService(FakeProvider(response))

    with pytest.raises(LLMProviderError):
        asyncio.run(service.chat("Hi"))
