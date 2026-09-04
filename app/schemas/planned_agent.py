from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.planning import ExecutionPlan


class PlannedAgentRunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=10_000)

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("goal must not be blank")
        return value


class PlannedAgentRunResponse(BaseModel):
    plan: ExecutionPlan
    status: Literal["completed", "approval_required"]
    current_step_index: int
    step_results: list[dict[str, Any]]
