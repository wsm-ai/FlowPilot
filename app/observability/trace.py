from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING

from app.observability.models import (
    AgentEvent,
    EventOutcome,
    EventStage,
    EventType,
)
if TYPE_CHECKING:
    from app.reliability.failures import FailureCategory


class TraceConfigurationError(ValueError):
    """Raised when trace data or recorder configuration is invalid."""


class TraceThreadAssociationConflictError(Exception):
    """Raised when a thread is already reserved by another trace."""


def _trace_identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TraceConfigurationError("trace_id must not be blank")
    return value.strip()


@dataclass(frozen=True, slots=True)
class TraceRecord:
    trace_id: str
    sequence: int
    event_type: EventType
    stage: EventStage
    outcome: EventOutcome
    occurred_at: datetime
    run_id: str | None
    thread_id: str | None
    failure_category: "FailureCategory | None"

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _trace_identifier(self.trace_id))
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 1
        ):
            raise TraceConfigurationError(
                "sequence must be a positive integer"
            )
        try:
            event = AgentEvent(
                event_type=self.event_type,
                occurred_at=self.occurred_at,
                stage=self.stage,
                outcome=self.outcome,
                run_id=self.run_id,
                thread_id=self.thread_id,
                failure_category=self.failure_category,
            )
        except ValueError as exc:
            raise TraceConfigurationError("trace event is invalid") from exc
        object.__setattr__(self, "occurred_at", event.occurred_at)
        object.__setattr__(self, "run_id", event.run_id)
        object.__setattr__(self, "thread_id", event.thread_id)

    @classmethod
    def from_event(
        cls,
        trace_id: str,
        sequence: int,
        event: AgentEvent,
    ) -> "TraceRecord":
        if not isinstance(event, AgentEvent):
            raise TraceConfigurationError("event must be an AgentEvent")
        return cls(
            trace_id=trace_id,
            sequence=sequence,
            event_type=event.event_type,
            stage=event.stage,
            outcome=event.outcome,
            occurred_at=event.occurred_at,
            run_id=event.run_id,
            thread_id=event.thread_id,
            failure_category=event.failure_category,
        )


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    trace_id: str
    records: tuple[TraceRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _trace_identifier(self.trace_id))
        if not isinstance(self.records, tuple):
            raise TraceConfigurationError("records must be an immutable tuple")
        previous = 0
        for record in self.records:
            if not isinstance(record, TraceRecord):
                raise TraceConfigurationError("trace record is invalid")
            if record.trace_id != self.trace_id:
                raise TraceConfigurationError(
                    "trace record belongs to another trace"
                )
            if record.sequence <= previous:
                raise TraceConfigurationError(
                    "trace record sequence must be strictly increasing"
                )
            previous = record.sequence


class InMemoryTraceRecorder:
    """Bounded process-local trace snapshots; not a durable audit store.

    When capacity eviction removes a trace's final retained record, its
    sequence state is also discarded. Reusing that trace ID starts a new
    in-memory retention cycle at sequence 1. Sequence continuity therefore
    is not guaranteed across eviction cycles or process restarts, and this
    recorder must not be used as a persistent audit source.
    """

    def __init__(self, *, capacity: int = 1_000) -> None:
        if (
            not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity < 1
        ):
            raise TraceConfigurationError("capacity must be a positive integer")
        self._capacity = capacity
        self._records: dict[str, list[TraceRecord]] = {}
        self._order: list[tuple[str, int]] = []
        self._next_sequence: dict[str, int] = {}
        self._lock = Lock()

    def record(self, trace_id: str, event: AgentEvent) -> TraceRecord:
        normalized = _trace_identifier(trace_id)
        if not isinstance(event, AgentEvent):
            raise TraceConfigurationError("event must be an AgentEvent")
        with self._lock:
            sequence = self._next_sequence.get(normalized, 1)
            record = TraceRecord.from_event(normalized, sequence, event)
            self._next_sequence[normalized] = sequence + 1
            self._records.setdefault(normalized, []).append(record)
            self._order.append((normalized, sequence))
            if len(self._order) > self._capacity:
                evicted_trace_id, evicted_sequence = self._order.pop(0)
                retained = self._records[evicted_trace_id]
                self._records[evicted_trace_id] = [
                    item
                    for item in retained
                    if item.sequence != evicted_sequence
                ]
                if not self._records[evicted_trace_id]:
                    del self._records[evicted_trace_id]
                    del self._next_sequence[evicted_trace_id]
            return record

    def get_trace(self, trace_id: str) -> ExecutionTrace:
        normalized = _trace_identifier(trace_id)
        with self._lock:
            records = tuple(self._records.get(normalized, ()))
        return ExecutionTrace(trace_id=normalized, records=records)
