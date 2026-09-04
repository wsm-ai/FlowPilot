import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.core.config import Settings
from app.providers.deepseek import DeepSeekProvider


def test_complete_maps_tool_calls_and_sends_tool_configuration():
    provider = DeepSeekProvider(Settings(deepseek_api_key="test-key"))
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_123",
                                function=SimpleNamespace(
                                    name="get_customer_feedback",
                                    arguments='{"customer_id":"C001"}',
                                ),
                            )
                        ],
                    )
                )
            ]
        )
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    tools = [{"type": "function", "function": {"name": "get_customer_feedback"}}]

    response = asyncio.run(
        provider.complete(
            messages=[{"role": "user", "content": "Find feedback"}],
            tools=tools,
        )
    )

    assert response.content is None
    assert response.tool_calls[0].id == "call_123"
    assert response.tool_calls[0].name == "get_customer_feedback"
    assert response.tool_calls[0].arguments == '{"customer_id":"C001"}'
    create.assert_awaited_once_with(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": "Find feedback"}],
        tools=tools,
        tool_choice="auto",
        extra_body={"thinking": {"type": "disabled"}},
    )
