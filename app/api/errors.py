from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.reliability.failures import (
    FailureCategory,
    FailureDescriptor,
    FailureDomain,
    classify_failure,
)
from app.grounding.evidence import EvidenceExtractionError
from app.mcp.client import MCPError
from app.persistence.repository import PersistenceError
from app.providers.base import LLMProviderError
from app.reliability.side_effects import SideEffectExecutionError
from app.reliability.timeouts import OperationTimeoutError
from app.services.grounded_answer_service import GroundedAnswerError
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError


class APIError(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    category: FailureCategory
    domain: FailureDomain


class APIErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: APIError


_STATUS_BY_CATEGORY = {
    FailureCategory.VALIDATION: 422,
    FailureCategory.BUSINESS_TERMINAL: status.HTTP_409_CONFLICT,
    FailureCategory.AMBIGUOUS_SIDE_EFFECT: status.HTTP_409_CONFLICT,
    FailureCategory.TRANSIENT: status.HTTP_503_SERVICE_UNAVAILABLE,
    FailureCategory.CONFIGURATION: status.HTTP_500_INTERNAL_SERVER_ERROR,
    FailureCategory.PERMANENT: status.HTTP_500_INTERNAL_SERVER_ERROR,
    # Defensive only: real task cancellation is not caught by these handlers.
    FailureCategory.CANCELLED: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def http_status_for_failure(failure: FailureDescriptor) -> int:
    return _STATUS_BY_CATEGORY.get(
        failure.category,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def response_for_failure(failure: FailureDescriptor) -> APIErrorResponse:
    return APIErrorResponse(
        error=APIError(
            code=failure.code,
            message=failure.safe_message,
            category=failure.category,
            domain=failure.domain,
        )
    )


def _json_error_response(failure: FailureDescriptor) -> JSONResponse:
    body = response_for_failure(failure)
    return JSONResponse(
        status_code=http_status_for_failure(failure),
        content=body.model_dump(mode="json"),
    )


async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    del request, exc
    return _json_error_response(
        FailureDescriptor(
            domain=FailureDomain.WORKFLOW,
            category=FailureCategory.VALIDATION,
            code="request_validation_error",
            safe_message="Request validation failed",
        )
    )


async def application_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    del request
    return _json_error_response(classify_failure(exc))


def register_api_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        RequestValidationError,
        request_validation_exception_handler,
    )
    for exception_type in (
        EvidenceExtractionError,
        MCPError,
        PersistenceError,
        LLMProviderError,
        SideEffectExecutionError,
        OperationTimeoutError,
        GroundedAnswerError,
        PlanningError,
        ToolExecutionError,
    ):
        app.add_exception_handler(exception_type, application_exception_handler)
    app.add_exception_handler(Exception, application_exception_handler)
