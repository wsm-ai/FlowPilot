import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_llm_service
from app.main import app
from app.providers.base import LLMProviderError


class FakeLLMService:
    model = "deepseek-v4-flash"

    async def chat(self, message: str) -> str:
        return f"Mock reply to: {message}"


def override_llm_service() -> FakeLLMService:
    return FakeLLMService()


@pytest.fixture
def client():
    app.dependency_overrides[get_llm_service] = override_llm_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_llm_service, None)


def test_chat_returns_reply(client: TestClient):
    response = client.post(
        "/api/v1/chat",
        json={"message": "What is an AI Agent?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "Mock reply to: What is an AI Agent?",
        "model": "deepseek-v4-flash",
    }


def test_chat_rejects_empty_message(client: TestClient):
    response = client.post("/api/v1/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_rejects_blank_message(client: TestClient):
    response = client.post("/api/v1/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_hides_provider_error(client: TestClient):
    class FailingLLMService:
        model = "deepseek-v4-flash"

        async def chat(self, message: str) -> str:
            raise LLMProviderError("sensitive third-party error")

    app.dependency_overrides[get_llm_service] = FailingLLMService

    response = client.post("/api/v1/chat", json={"message": "Hello"})

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The language model service is unavailable"
    }
    assert "sensitive third-party error" not in response.text
