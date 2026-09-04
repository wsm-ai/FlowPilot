import asyncio
from dataclasses import replace

import pytest

from app.persistence.repository import PersistenceError
from app.providers.base import LLMProviderError
from app.services.graph_agent_service import GraphAgentResult, GraphExecutedTool
from app.services.persistent_agent_service import PersistentAgentService
from app.tools.base import ToolExecutionError


class FakeRunRepository:
    def __init__(self, *, fail_create: bool = False, fail_update: bool = False):
        self.fail_create = fail_create
        self.fail_update = fail_update
        self.created = []
        self.updates = []

    async def create(self, record):
        if self.fail_create:
            raise PersistenceError("create failed")
        self.created.append(record)

    async def update(self, run_id, *, status, result=None, error_type=None):
        if self.fail_update:
            raise PersistenceError("update failed")
        self.updates.append(
            {
                "run_id": run_id,
                "status": status,
                "result": result,
                "error_type": error_type,
            }
        )
        return replace(
            self.created[0],
            status=status,
            result=result,
            error_type=error_type,
        )


class SuccessfulAgent:
    async def run(self, message: str, *, thread_id: str | None = None):
        return GraphAgentResult(
            answer="Done",
            executed_tools=[
                GraphExecutedTool(
                    tool_call_id="call_123",
                    name="get_customer_feedback",
                    arguments={"customer_id": "C001"},
                )
            ],
        )


def test_successful_run_is_created_then_completed_with_safe_result():
    repository = FakeRunRepository()
    service = PersistentAgentService(SuccessfulAgent(), repository)

    result = asyncio.run(service.run("Review feedback", thread_id="thread-a"))

    assert len(repository.created) == 1
    created = repository.created[0]
    assert created.status == "running"
    assert created.mode == "reactive"
    assert created.input_text == "Review feedback"
    assert result.run_id == created.run_id
    assert result.answer == "Done"
    assert repository.updates == [
        {
            "run_id": created.run_id,
            "status": "completed",
            "result": {
                "answer": "Done",
                "executed_tools": [
                    {
                        "tool_call_id": "call_123",
                        "name": "get_customer_feedback",
                        "arguments": {"customer_id": "C001"},
                    }
                ],
            },
            "error_type": None,
        }
    ]
    assert "messages" not in repository.updates[0]["result"]


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (LLMProviderError("sensitive provider detail"), "LLMProviderError"),
        (ToolExecutionError("sensitive tool detail"), "ToolExecutionError"),
    ],
)
def test_agent_failure_is_recorded_by_type_and_reraised(error, error_type):
    class FailingAgent:
        async def run(self, message: str, *, thread_id: str | None = None):
            raise error

    repository = FakeRunRepository()
    service = PersistentAgentService(FailingAgent(), repository)

    with pytest.raises(type(error)) as raised:
        asyncio.run(service.run("Fail"))

    assert raised.value is error
    assert repository.updates[0]["status"] == "failed"
    assert repository.updates[0]["error_type"] == error_type
    assert repository.updates[0]["result"] is None
    assert str(error) not in repository.updates[0]["error_type"]


def test_repository_create_failure_propagates_without_update_attempt():
    repository = FakeRunRepository(fail_create=True)
    service = PersistentAgentService(SuccessfulAgent(), repository)

    with pytest.raises(PersistenceError):
        asyncio.run(service.run("Hello"))

    assert repository.updates == []


def test_completed_update_failure_propagates():
    repository = FakeRunRepository(fail_update=True)
    service = PersistentAgentService(SuccessfulAgent(), repository)

    with pytest.raises(PersistenceError):
        asyncio.run(service.run("Hello"))
