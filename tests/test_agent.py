import pytest
from fastapi.testclient import TestClient

from app.api.agent import get_persistent_agent_service
from app.main import app
from app.persistence.repository import PersistenceError
from app.providers.base import LLMProviderError
from app.services.graph_agent_service import (
    GraphExecutedTool,
)
from app.services.persistent_agent_service import PersistentAgentResult
from app.tools.base import ToolExecutionError


class FakeGraphAgentService:
    async def run(
        self,
        message: str,
        *,
        thread_id: str | None = None,
    ) -> PersistentAgentResult:
        return PersistentAgentResult(
            run_id="run-123",
            answer="Customer C001 has two high-priority issues.",
            executed_tools=[
                GraphExecutedTool(
                    tool_call_id="call_123",
                    name="get_customer_feedback",
                    arguments={"customer_id": "C001", "priority": "high"},
                )
            ],
        )


def override_graph_agent_service() -> FakeGraphAgentService:
    return FakeGraphAgentService()


@pytest.fixture
def client():
    app.dependency_overrides[
        get_persistent_agent_service
    ] = override_graph_agent_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_persistent_agent_service, None)


def test_agent_returns_answer_and_executed_tools(client: TestClient):
    response = client.post(
        "/api/v1/agent/run",
        json={"message": "Find high-priority feedback for C001"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-123",
        "answer": "Customer C001 has two high-priority issues.",
        "executed_tools": [
            {
                "tool_call_id": "call_123",
                "name": "get_customer_feedback",
                "arguments": {"customer_id": "C001", "priority": "high"},
            }
        ],
    }


def test_agent_returns_direct_answer_without_executed_tools(client: TestClient):
    class DirectAnswerService:
        async def run(
            self,
            message: str,
            *,
            thread_id: str | None = None,
        ) -> PersistentAgentResult:
            return PersistentAgentResult(run_id="run-direct", answer="Direct answer")

    app.dependency_overrides[get_persistent_agent_service] = DirectAnswerService

    response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "run_id": "run-direct",
        "answer": "Direct answer",
        "executed_tools": [],
    }


@pytest.mark.parametrize("message", ["", "   "])
def test_agent_rejects_empty_or_blank_message(client: TestClient, message: str):
    response = client.post("/api/v1/agent/run", json={"message": message})

    assert response.status_code == 422


@pytest.mark.parametrize("thread_id", ["", "   ", "x" * 201])
def test_agent_rejects_invalid_thread_id(client: TestClient, thread_id: str):
    response = client.post(
        "/api/v1/agent/run",
        json={"message": "Hello", "thread_id": thread_id},
    )

    assert response.status_code == 422


def test_agent_accepts_and_forwards_thread_id(client: TestClient):
    received_thread_ids = []

    class ThreadAwareService:
        async def run(self, message: str, *, thread_id: str | None = None):
            received_thread_ids.append(thread_id)
            return PersistentAgentResult(run_id="run-thread", answer="Answer")

    app.dependency_overrides[get_persistent_agent_service] = ThreadAwareService

    response = client.post(
        "/api/v1/agent/run",
        json={"message": "Hello", "thread_id": "thread-123"},
    )

    assert response.status_code == 200
    assert received_thread_ids == ["thread-123"]


def test_agent_hides_provider_error(client: TestClient):
    class FailingProviderService:
        async def run(self, message: str, *, thread_id: str | None = None):
            raise LLMProviderError("sensitive provider error")

    app.dependency_overrides[get_persistent_agent_service] = FailingProviderService

    response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The language model service is unavailable"
    }
    assert "sensitive provider error" not in response.text


def test_agent_hides_tool_error(client: TestClient):
    class FailingToolService:
        async def run(self, message: str, *, thread_id: str | None = None):
            raise ToolExecutionError("sensitive tool error")

    app.dependency_overrides[get_persistent_agent_service] = FailingToolService

    response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Tool execution failed"}
    assert "sensitive tool error" not in response.text


def test_agent_hides_persistence_error(client: TestClient):
    class FailingPersistenceService:
        async def run(self, message: str, *, thread_id: str | None = None):
            raise PersistenceError("sensitive sqlite error")

    app.dependency_overrides[
        get_persistent_agent_service
    ] = FailingPersistenceService

    response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Agent persistence failed"}
    assert "sensitive sqlite error" not in response.text
