from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_llm_service
from app.providers.base import LLMProviderError
from app.schemas.agent import AgentRunRequest, AgentRunResponse, ExecutedToolResponse
from app.services.llm_service import LLMService
from app.services.tool_calling_service import ToolCallingService
from app.tools.base import ToolExecutionError
from app.tools.registry import create_default_tool_registry


router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


def get_tool_calling_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> ToolCallingService:
    return ToolCallingService(
        llm_service=llm_service,
        registry=create_default_tool_registry(),
    )


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    service: ToolCallingService = Depends(get_tool_calling_service),
) -> AgentRunResponse:
    try:
        result = await service.run(request.message)
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model service is unavailable",
        ) from exc
    except ToolExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tool execution failed",
        ) from exc

    return AgentRunResponse(
        answer=result.answer,
        executed_tools=[
            ExecutedToolResponse(
                tool_call_id=tool.tool_call_id,
                name=tool.name,
                arguments=tool.arguments,
            )
            for tool in result.executed_tools
        ],
    )
