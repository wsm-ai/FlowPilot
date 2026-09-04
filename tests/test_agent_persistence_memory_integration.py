import asyncio
from typing import Any

from httpx import ASGITransport, AsyncClient

from app.api.agent import get_persistent_agent_service
from app.main import app
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.providers.types import LLMResponse
from app.services.graph_agent_service import GraphAgentService
from app.services.llm_service import LLMService
from app.services.persistent_agent_service import PersistentAgentService
from app.tools.registry import create_default_tool_registry


class FakeProvider:
    model = "test-model"

    def __init__(self) -> None:
        self._responses = iter(
            [
                LLMResponse(content="Got it."),
                LLMResponse(content="You mentioned C001."),
            ]
        )
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


def test_api_thread_memory_and_run_persistence_are_independent(tmp_path):
    run_database = tmp_path / "runs.db"
    checkpoint_database = tmp_path / "checkpoints.sqlite"
    provider = FakeProvider()

    async def scenario():
        repository = SQLiteRunRepository(run_database)
        await repository.initialize()
        async with async_checkpoint_saver(checkpoint_database) as saver:
            graph_service = GraphAgentService(
                llm_service=LLMService(provider),
                registry=create_default_tool_registry(),
                checkpointer=saver,
            )
            persistent_service = PersistentAgentService(graph_service, repository)
            app.dependency_overrides[
                get_persistent_agent_service
            ] = lambda: persistent_service
            try:
                async with AsyncClient(
                    transport=ASGITransport(app=app),
                    base_url="http://test",
                ) as client:
                    first = await client.post(
                        "/api/v1/agent/run",
                        json={
                            "message": "My customer is C001.",
                            "thread_id": "thread-api",
                        },
                    )
                    second = await client.post(
                        "/api/v1/agent/run",
                        json={
                            "message": "What customer did I mention?",
                            "thread_id": "thread-api",
                        },
                    )
            finally:
                app.dependency_overrides.pop(get_persistent_agent_service, None)

        return first, second, await repository.list_recent()

    first, second, runs = asyncio.run(scenario())

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["run_id"] != second.json()["run_id"]
    assert provider.requests[1]["messages"] == [
        {"role": "user", "content": "My customer is C001."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "What customer did I mention?"},
    ]
    assert len(runs) == 2
    assert all(run.status == "completed" for run in runs)
