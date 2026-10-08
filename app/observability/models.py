from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.reliability.failures import FailureCategory


class ObservabilityConfigurationError(ValueError):
    """Raised when a structured event is invalid."""


class EventType(StrEnum):
    PLANNER_STARTED = "planner_started"
    PLANNER_COMPLETED = "planner_completed"
    PLANNER_FAILED = "planner_failed"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_APPROVED = "approval_approved"
    APPROVAL_REJECTED = "approval_rejected"
    AGENT_RUN_STARTED = "agent_run_started"
    AGENT_RUN_COMPLETED = "agent_run_completed"
    AGENT_RUN_FAILED = "agent_run_failed"


class EventStage(StrEnum):
    PLANNER = "planner"
    AGENT = "agent"
    APPROVAL = "approval"
    TOOL = "tool"
    RELIABILITY = "reliability"


class EventOutcome(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    WAITING = "waiting"
    APPROVED = "approved"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


_EVENT_SEMANTICS = {
    EventType.PLANNER_STARTED: (EventStage.PLANNER, EventOutcome.STARTED),
    EventType.PLANNER_COMPLETED: (
        EventStage.PLANNER,
        EventOutcome.SUCCEEDED,
    ),
    EventType.PLANNER_FAILED: (EventStage.PLANNER, EventOutcome.FAILED),
    EventType.APPROVAL_REQUIRED: (
        EventStage.APPROVAL,
        EventOutcome.WAITING,
    ),
    EventType.APPROVAL_APPROVED: (
        EventStage.APPROVAL,
        EventOutcome.APPROVED,
    ),
    EventType.APPROVAL_REJECTED: (
        EventStage.APPROVAL,
        EventOutcome.REJECTED,
    ),
    EventType.AGENT_RUN_STARTED: (EventStage.AGENT, EventOutcome.STARTED),
    EventType.AGENT_RUN_COMPLETED: (
        EventStage.AGENT,
        EventOutcome.SUCCEEDED,
    ),
    EventType.AGENT_RUN_FAILED: (EventStage.AGENT, EventOutcome.FAILED),
}


def _optional_identifier(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ObservabilityConfigurationError(f"{name} must not be blank")
    return value.strip()


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_type: EventType
    occurred_at: datetime
    stage: EventStage
    outcome: EventOutcome
    run_id: str | None = None
    thread_id: str | None = None
    failure_category: "FailureCategory | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, EventType):
            raise ObservabilityConfigurationError("event_type is invalid")
        if not isinstance(self.stage, EventStage):
            raise ObservabilityConfigurationError("stage is invalid")
        if not isinstance(self.outcome, EventOutcome):
            raise ObservabilityConfigurationError("outcome is invalid")
        if _EVENT_SEMANTICS[self.event_type] != (self.stage, self.outcome):
            raise ObservabilityConfigurationError(
                "event type, stage, and outcome are inconsistent"
            )
        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
        ):
            raise ObservabilityConfigurationError(
                "occurred_at must be timezone-aware"
            )
        if self.failure_category is not None:
            from app.reliability.failures import FailureCategory

            if not isinstance(self.failure_category, FailureCategory):
                raise ObservabilityConfigurationError(
                    "failure_category is invalid"
                )
        object.__setattr__(
            self, "occurred_at", self.occurred_at.astimezone(timezone.utc)
        )
        object.__setattr__(
            self, "run_id", _optional_identifier(self.run_id, "run_id")
        )
        object.__setattr__(
            self,
            "thread_id",
            _optional_identifier(self.thread_id, "thread_id"),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
