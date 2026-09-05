from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.planning import ExecutionPlan


class PlannedAgentRunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=10_000)
    thread_id: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("goal must not be blank")
        return value

    @field_validator("thread_id")
    @classmethod
    def thread_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("thread_id must not be blank")
        return value


class CitationResponse(BaseModel):
    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    title: str | None = None


class PlannedAgentRunResponse(BaseModel):
    run_id: str
    thread_id: str
    plan: ExecutionPlan
    status: Literal["completed", "approval_required", "rejected"]
    current_step_index: int
    step_results: list[dict[str, Any]]
    pending_approval: dict[str, Any] | None
    answer: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    answer_status: Literal["not_attempted", "completed", "failed"] = "not_attempted"


class ApprovalResumeRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)
    decision: Literal["approve", "reject"]

    @field_validator("run_id", "thread_id")
    @classmethod
    def identifier_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value


class GroundedAnswerRetryRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=200)

    @field_validator("run_id", "thread_id")
    @classmethod
    def identifier_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value
