from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.providers.base import LLMProviderError
from app.providers.deepseek import DeepSeekProvider
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService


router = APIRouter(prefix="/api/v1", tags=["Chat"])


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    return LLMService(DeepSeekProvider(settings))


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    try:
        reply = await service.chat(request.message)
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model service is unavailable",
        ) from exc

    return ChatResponse(reply=reply, model=service.model)
