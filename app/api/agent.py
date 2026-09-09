from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.api.dependencies import (
    get_checkpointer,
    get_llm_service,
    get_mcp_approval_required_actions,
    get_reactive_tool_registry,
    get_run_repository,
    get_side_effect_executor,
    get_tool_registry,
)
from app.persistence.repository import RunRepository
from app.reliability.side_effects import SideEffectExecutor
from app.schemas.agent import AgentRunRequest, AgentRunResponse, ExecutedToolResponse
from app.schemas.planned_agent import (
    ApprovalResumeRequest,
    CitationResponse,
    GroundedAnswerRetryRequest,
    PlannedAgentRunRequest,
    PlannedAgentRunResponse,
)
from app.services.grounded_answer_service import GroundedAnswerService
from app.services.approval_workflow_service import (
    ApprovalNotPendingError,
    ApprovalThreadConflictError,
    ApprovalThreadNotFoundError,
    ApprovalWorkflowService,
)
from app.services.graph_agent_service import GraphAgentService
from app.services.llm_service import LLMService
from app.services.planned_agent_service import PlannedAgentService
from app.services.planner_service import PlannerService
from app.services.persistent_agent_service import PersistentAgentService
from app.services.persistent_approval_workflow_service import (
    ApprovalRunNotFoundError,
    ApprovalRunNotPendingError,
    ApprovalRunThreadMismatchError,
    GroundedAnswerRetryNotAllowedError,
    GroundedAnswerRetryStateError,
    PersistentApprovalWorkflowResult,
    PersistentApprovalWorkflowService,
)
from app.tools.registry import ToolRegistry


router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


def get_graph_agent_service(
    llm_service: LLMService = Depends(get_llm_service),
    checkpointer: BaseCheckpointSaver = Depends(get_checkpointer),
    registry: ToolRegistry = Depends(get_reactive_tool_registry),
) -> GraphAgentService:
    return GraphAgentService(
        llm_service=llm_service,
        registry=registry,
        checkpointer=checkpointer,
    )


def get_persistent_agent_service(
    graph_agent_service: GraphAgentService = Depends(get_graph_agent_service),
    run_repository: RunRepository = Depends(get_run_repository),
) -> PersistentAgentService:
    return PersistentAgentService(graph_agent_service, run_repository)


def get_planned_agent_service(
    llm_service: LLMService = Depends(get_llm_service),
    registry: ToolRegistry = Depends(get_tool_registry),
    approval_required_actions: frozenset[str] = Depends(
        get_mcp_approval_required_actions
    ),
) -> PlannedAgentService:
    return PlannedAgentService(
        planner_service=PlannerService(
            llm_service,
            tool_definitions=registry.definitions(),
            approval_required_actions=approval_required_actions,
        ),
        registry=registry,
        grounded_answer_service=GroundedAnswerService(llm_service),
    )


def get_approval_workflow_service(
    llm_service: LLMService = Depends(get_llm_service),
    checkpointer: BaseCheckpointSaver = Depends(get_checkpointer),
    registry: ToolRegistry = Depends(get_tool_registry),
    approval_required_actions: frozenset[str] = Depends(
        get_mcp_approval_required_actions
    ),
    side_effect_executor: SideEffectExecutor = Depends(get_side_effect_executor),
) -> ApprovalWorkflowService:
    return ApprovalWorkflowService(
        planner_service=PlannerService(
            llm_service,
            tool_definitions=registry.definitions(),
            approval_required_actions=approval_required_actions,
        ),
        registry=registry,
        checkpointer=checkpointer,
        grounded_answer_service=GroundedAnswerService(llm_service),
        side_effect_executor=side_effect_executor,
    )


def get_persistent_approval_workflow_service(
    workflow_service: ApprovalWorkflowService = Depends(
        get_approval_workflow_service
    ),
    run_repository: RunRepository = Depends(get_run_repository),
) -> PersistentApprovalWorkflowService:
    return PersistentApprovalWorkflowService(workflow_service, run_repository)


def _planned_agent_response(
    result: PersistentApprovalWorkflowResult,
) -> PlannedAgentRunResponse:
    grounded_answer = (
        result.grounded_answer
        if result.grounding_status == "completed"
        else None
    )
    return PlannedAgentRunResponse(
        run_id=result.run_id,
        thread_id=result.thread_id,
        plan=result.plan,
        status=result.status,
        current_step_index=result.current_step_index,
        step_results=result.step_results,
        pending_approval=result.pending_approval,
        answer=None if grounded_answer is None else grounded_answer.answer,
        citations=(
            []
            if grounded_answer is None
            else [
                CitationResponse.model_validate(citation.model_dump())
                for citation in grounded_answer.citations
            ]
        ),
        answer_status=result.grounding_status,
    )


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    service: PersistentAgentService = Depends(get_persistent_agent_service),
) -> AgentRunResponse:
    result = await service.run(request.message, thread_id=request.thread_id)

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
    service: PersistentApprovalWorkflowService = Depends(
        get_persistent_approval_workflow_service
    ),
) -> PlannedAgentRunResponse:
    try:
        result = await service.start(request.goal, thread_id=request.thread_id)
    except ApprovalThreadConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Approval thread already exists",
        ) from exc

    return _planned_agent_response(result)


@router.post("/approval/resume", response_model=PlannedAgentRunResponse)
async def resume_planned_agent(
    request: ApprovalResumeRequest,
    service: PersistentApprovalWorkflowService = Depends(
        get_persistent_approval_workflow_service
    ),
) -> PlannedAgentRunResponse:
    try:
        result = await service.resume(
            request.run_id,
            request.thread_id,
            request.decision,
        )
    except ApprovalRunNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval run not found",
        ) from exc
    except ApprovalRunThreadMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run does not match approval thread",
        ) from exc
    except ApprovalRunNotPendingError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Approval run is not pending",
        ) from exc
    except ApprovalThreadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval thread not found",
        ) from exc
    except ApprovalThreadConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Approval thread already exists",
        ) from exc
    except ApprovalNotPendingError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No pending approval for this thread",
        ) from exc

    return _planned_agent_response(result)


@router.post("/answer/retry", response_model=PlannedAgentRunResponse)
async def retry_grounded_answer(
    request: GroundedAnswerRetryRequest,
    service: PersistentApprovalWorkflowService = Depends(
        get_persistent_approval_workflow_service
    ),
) -> PlannedAgentRunResponse:
    try:
        result = await service.retry_grounded_answer(
            request.run_id,
            request.thread_id,
        )
    except ApprovalRunNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval run not found",
        ) from exc
    except ApprovalRunThreadMismatchError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run does not match approval thread",
        ) from exc
    except GroundedAnswerRetryNotAllowedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Grounded answer retry is not allowed",
        ) from exc
    except GroundedAnswerRetryStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Grounded answer retry state is invalid",
        ) from exc

    return _planned_agent_response(result)
