import asyncio
from typing import Any

import pytest

from app.graph.workflow import create_basic_agent_graph
from app.persistence.checkpoint import async_checkpoint_saver
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


def create_service(provider: FakeProvider, checkpointer=None) -> GraphAgentService:
    return GraphAgentService(
        llm_service=LLMService(provider),
        registry=create_default_tool_registry(),
        checkpointer=checkpointer,
    )


def test_same_thread_accumulates_complete_conversation_without_duplicates(tmp_path):
    database_path = tmp_path / "memory.sqlite"
    provider = FakeProvider(
        [
            LLMResponse(content="Got it."),
            LLMResponse(content="You mentioned C001."),
        ]
    )

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver:
            service = create_service(provider, saver)
            await service.run("My customer is C001.", thread_id="thread-a")
            await service.run(
                "What customer did I mention?",
                thread_id="thread-a",
            )
            inspection_graph = create_basic_agent_graph(
                LLMService(provider),
                create_default_tool_registry(),
                checkpointer=saver,
            )
            return await inspection_graph.aget_state(
                {"configurable": {"thread_id": "thread-a"}}
            )

    snapshot = asyncio.run(scenario())
    expected_messages = [
        {"role": "user", "content": "My customer is C001."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "What customer did I mention?"},
        {"role": "assistant", "content": "You mentioned C001."},
    ]

    assert provider.requests[1]["messages"] == expected_messages[:-1]
    assert snapshot.values["messages"] == expected_messages
    assert len(snapshot.values["messages"]) == 4


def test_different_threads_do_not_share_messages(tmp_path):
    database_path = tmp_path / "memory.sqlite"
    provider = FakeProvider(
        [
            LLMResponse(content="Answer A"),
            LLMResponse(content="Answer B"),
        ]
    )

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver:
            service = create_service(provider, saver)
            await service.run("Message A", thread_id="thread-a")
            await service.run("Message B", thread_id="thread-b")

    asyncio.run(scenario())

    assert provider.requests[0]["messages"] == [
        {"role": "user", "content": "Message A"}
    ]
    assert provider.requests[1]["messages"] == [
        {"role": "user", "content": "Message B"}
    ]


def test_tool_calling_history_is_available_in_next_round(tmp_path):
    database_path = tmp_path / "memory.sqlite"
    provider = FakeProvider(
        [
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
            LLMResponse(content="C001 has two high-priority issues."),
            LLMResponse(content="The customer was C001."),
        ]
    )

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver:
            service = create_service(provider, saver)
            await service.run("Find C001 feedback", thread_id="thread-tools")
            await service.run("Which customer was that?", thread_id="thread-tools")

    asyncio.run(scenario())

    second_round_messages = provider.requests[2]["messages"]
    assert [message["role"] for message in second_round_messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert second_round_messages[1]["tool_calls"][0]["id"] == "call_123"
    assert second_round_messages[2]["tool_call_id"] == "call_123"
    assert "FB-001" in second_round_messages[2]["content"]
    assert len(second_round_messages) == 5


def test_memory_survives_saver_reconnection(tmp_path):
    database_path = tmp_path / "memory.sqlite"

    async def scenario():
        provider_a = FakeProvider([LLMResponse(content="Got it.")])
        async with async_checkpoint_saver(database_path) as saver_a:
            await create_service(provider_a, saver_a).run(
                "My customer is C001.",
                thread_id="thread-durable",
            )

        provider_b = FakeProvider([LLMResponse(content="You mentioned C001.")])
        async with async_checkpoint_saver(database_path) as saver_b:
            await create_service(provider_b, saver_b).run(
                "What customer did I mention?",
                thread_id="thread-durable",
            )
        return provider_b.requests[0]["messages"]

    messages = asyncio.run(scenario())

    assert messages == [
        {"role": "user", "content": "My customer is C001."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "What customer did I mention?"},
    ]


def test_reactive_agent_without_checkpointer_remains_compatible():
    provider = FakeProvider([LLMResponse(content="Direct answer")])
    service = create_service(provider)

    result = asyncio.run(service.run("Hello"))

    assert result.answer == "Direct answer"
    assert provider.requests[0]["messages"] == [
        {"role": "user", "content": "Hello"}
    ]


def test_checkpointed_service_requires_thread_id(tmp_path):
    database_path = tmp_path / "memory.sqlite"
    provider = FakeProvider([LLMResponse(content="Answer")])

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver:
            await create_service(provider, saver).run("Hello")

    with pytest.raises(ValueError):
        asyncio.run(scenario())
