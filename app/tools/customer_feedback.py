from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.tools.base import ToolExecutionError


Priority = Literal["low", "medium", "high"]


class CustomerFeedbackArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(
        min_length=1,
        description="Customer identifier",
    )
    priority: Priority | None = Field(
        default=None,
        description="Optional feedback priority",
    )

    @field_validator("customer_id")
    @classmethod
    def customer_id_must_not_be_blank(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("customer_id must not be blank")
        return normalized_value


# DEMO DATA: temporary in-memory records used until a persistent data source exists.
DEMO_CUSTOMER_FEEDBACK: list[dict[str, str]] = [
    {
        "id": "FB-001",
        "customer_id": "C001",
        "priority": "high",
        "category": "bug",
        "content": "Login API intermittently returns HTTP 500.",
    },
    {
        "id": "FB-002",
        "customer_id": "C001",
        "priority": "medium",
        "category": "performance",
        "content": "Knowledge search response is slower than expected.",
    },
    {
        "id": "FB-003",
        "customer_id": "C001",
        "priority": "high",
        "category": "bug",
        "content": "File upload occasionally fails for PDF documents.",
    },
    {
        "id": "FB-004",
        "customer_id": "C002",
        "priority": "low",
        "category": "ui",
        "content": "Dashboard button alignment is inconsistent.",
    },
    {
        "id": "FB-005",
        "customer_id": "C002",
        "priority": "high",
        "category": "bug",
        "content": "Exported report sometimes contains duplicated rows.",
    },
]


class CustomerFeedbackTool:
    name = "get_customer_feedback"
    description = "Get customer feedback records by customer ID and optional priority."

    @property
    def parameters(self) -> dict[str, Any]:
        return CustomerFeedbackArguments.model_json_schema()

    async def execute(self, arguments: dict[str, Any]) -> list[dict[str, str]]:
        try:
            validated = CustomerFeedbackArguments.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError("Invalid arguments for get_customer_feedback") from exc

        return [
            record.copy()
            for record in DEMO_CUSTOMER_FEEDBACK
            if record["customer_id"] == validated.customer_id
            and (validated.priority is None or record["priority"] == validated.priority)
        ]
