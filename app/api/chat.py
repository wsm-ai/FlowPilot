from fastapi import APIRouter, Depends

from app.api.dependencies import get_llm_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService


router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    reply = await service.chat(request.message)

    return ChatResponse(reply=reply, model=service.model)
