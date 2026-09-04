from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.api.dependencies import (
    get_checkpointer,
    get_llm_service,
    get_run_repository,
)
from app.persistence.repository import PersistenceError, RunRepository
from app.providers.base import LLMProviderError
from app.schemas.agent import AgentRunRequest, AgentRunResponse, ExecutedToolResponse
from app.schemas.planned_agent import PlannedAgentRunRequest, PlannedAgentRunResponse
from app.services.graph_agent_service import GraphAgentService
from app.services.llm_service import LLMService
from app.services.planned_agent_service import PlannedAgentService
from app.services.planner_service import PlannerService, PlanningError
from app.services.persistent_agent_service import PersistentAgentService
from app.tools.base import ToolExecutionError
from app.tools.registry import create_default_tool_registry


router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


def get_graph_agent_service(
    llm_service: LLMService = Depends(get_llm_service),
    checkpointer: BaseCheckpointSaver = Depends(get_checkpointer),
) -> GraphAgentService:
    return GraphAgentService(
        llm_service=llm_service,
        registry=create_default_tool_registry(),
        checkpointer=checkpointer,
    )


def get_persistent_agent_service(
    graph_agent_service: GraphAgentService = Depends(get_graph_agent_service),
    run_repository: RunRepository = Depends(get_run_repository),
) -> PersistentAgentService:
    return PersistentAgentService(graph_agent_service, run_repository)


def get_planned_agent_service(
    llm_service: LLMService = Depends(get_llm_service),
) -> PlannedAgentService:
    return PlannedAgentService(
        planner_service=PlannerService(llm_service),
        registry=create_default_tool_registry(),
    )


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    service: PersistentAgentService = Depends(get_persistent_agent_service),
) -> AgentRunResponse:
    try:
        result = await service.run(request.message, thread_id=request.thread_id)
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
    except PersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent persistence failed",
        ) from exc

    return AgentRunResponse(
        run_id=result.run_id,
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


@router.post("/plan-run", response_model=PlannedAgentRunResponse)
async def run_planned_agent(
    request: PlannedAgentRunRequest,
    service: PlannedAgentService = Depends(get_planned_agent_service),
) -> PlannedAgentRunResponse:
    try:
        result = await service.run(request.goal)
    except PlanningError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to execute the requested plan",
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model service is unavailable",
        ) from exc
    except ToolExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Plan step execution failed",
        ) from exc

    return PlannedAgentRunResponse(
        plan=result.plan,
        status=result.status,
        current_step_index=result.current_step_index,
        step_results=result.step_results,
    )
