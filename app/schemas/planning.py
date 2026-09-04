from pydantic import BaseModel, Field, field_validator


class PlanStep(BaseModel):
    id: int = Field(gt=0)
    description: str = Field(min_length=1)
    action: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    )
    requires_approval: bool = False

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("description must not be blank")
        return normalized

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ExecutionPlan(BaseModel):
    goal: str = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("goal must not be blank")
        return normalized
