from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_llm_service
from app.providers.base import LLMProviderError
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService


router = APIRouter(prefix="/api/v1", tags=["Chat"])


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
