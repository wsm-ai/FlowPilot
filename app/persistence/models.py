from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


RunMode = Literal["reactive", "planned"]
RunStatus = Literal["running", "completed", "approval_required", "failed"]


@dataclass(slots=True)
class AgentRunRecord:
    run_id: str
    mode: RunMode
    input_text: str
    status: RunStatus
    result: dict[str, Any] | None
    error_type: str | None
    created_at: datetime
    updated_at: datetime


def create_agent_run_record(mode: RunMode, input_text: str) -> AgentRunRecord:
    now = datetime.now(timezone.utc)
    return AgentRunRecord(
        run_id=str(uuid4()),
        mode=mode,
        input_text=input_text,
        status="running",
        result=None,
        error_type=None,
        created_at=now,
        updated_at=now,
    )
