import pytest
from fastapi.testclient import TestClient

from app.api.agent import get_tool_calling_service
from app.main import app
from app.providers.base import LLMProviderError
from app.services.tool_calling_service import ExecutedTool, ToolCallingResult
from app.tools.base import ToolExecutionError


class FakeToolCallingService:
    async def run(self, message: str) -> ToolCallingResult:
        return ToolCallingResult(
            answer="Customer C001 has two high-priority issues.",
            executed_tools=[
                ExecutedTool(
                    tool_call_id="call_123",
                    name="get_customer_feedback",
                    arguments={"customer_id": "C001", "priority": "high"},
                )
            ],
        )


def override_tool_calling_service() -> FakeToolCallingService:
    return FakeToolCallingService()


@pytest.fixture
def client():
    app.dependency_overrides[
        get_tool_calling_service
    ] = override_tool_calling_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_tool_calling_service, None)


def test_agent_returns_answer_and_executed_tools(client: TestClient):
    response = client.post(
        "/api/v1/agent/run",
        json={"message": "Find high-priority feedback for C001"},
    )

    assert response.status_code == 200
    assert response.json() == {
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
        async def run(self, message: str) -> ToolCallingResult:
            return ToolCallingResult(answer="Direct answer")

    app.dependency_overrides[get_tool_calling_service] = DirectAnswerService

    response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Direct answer",
        "executed_tools": [],
    }


@pytest.mark.parametrize("message", ["", "   "])
def test_agent_rejects_empty_or_blank_message(client: TestClient, message: str):
    response = client.post("/api/v1/agent/run", json={"message": message})

    assert response.status_code == 422


def test_agent_hides_provider_error(client: TestClient):
    class FailingProviderService:
        async def run(self, message: str) -> ToolCallingResult:
            raise LLMProviderError("sensitive provider error")

    app.dependency_overrides[get_tool_calling_service] = FailingProviderService

    response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The language model service is unavailable"
    }
    assert "sensitive provider error" not in response.text


def test_agent_hides_tool_error(client: TestClient):
    class FailingToolService:
        async def run(self, message: str) -> ToolCallingResult:
            raise ToolExecutionError("sensitive tool error")

    app.dependency_overrides[get_tool_calling_service] = FailingToolService

    response = client.post("/api/v1/agent/run", json={"message": "Hello"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Tool execution failed"}
    assert "sensitive tool error" not in response.text
