from dataclasses import FrozenInstanceError, fields
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from app.observability import (
    AgentEvent,
    EventOutcome,
    EventStage,
    EventType,
    ExecutionTrace,
    InMemoryTraceRecorder,
    TraceConfigurationError,
    TraceRecord,
)
from app.reliability.failures import FailureCategory


FIXED_TIME = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


def _event(
    event_type: EventType = EventType.PLANNER_STARTED,
    *,
    occurred_at: datetime = FIXED_TIME,
    run_id: str | None = None,
    thread_id: str | None = None,
    failure_category: FailureCategory | None = None,
) -> AgentEvent:
    semantics = {
        EventType.PLANNER_STARTED: (
            EventStage.PLANNER, EventOutcome.STARTED
        ),
        EventType.PLANNER_COMPLETED: (
            EventStage.PLANNER, EventOutcome.SUCCEEDED
        ),
        EventType.PLANNER_FAILED: (
            EventStage.PLANNER, EventOutcome.FAILED
        ),
    }
    stage, outcome = semantics[event_type]
    return AgentEvent(
        event_type=event_type,
        occurred_at=occurred_at,
        stage=stage,
        outcome=outcome,
        run_id=run_id,
        thread_id=thread_id,
        failure_category=failure_category,
    )


def _record(
    trace_id: str = "trace-a",
    sequence: int = 1,
) -> TraceRecord:
    return TraceRecord.from_event(trace_id, sequence, _event())


def test_trace_record_and_execution_trace_are_immutable() -> None:
    record = _record()
    trace = ExecutionTrace("trace-a", (record,))
    with pytest.raises(FrozenInstanceError):
        record.sequence = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        trace.records = ()  # type: ignore[misc]


@pytest.mark.parametrize("trace_id", ["", "   ", None, 7])
def test_invalid_trace_id_is_rejected(trace_id: object) -> None:
    with pytest.raises(TraceConfigurationError, match="trace_id"):
        InMemoryTraceRecorder().get_trace(trace_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("sequence", [0, -1, True, 1.5])
def test_invalid_sequence_is_rejected(sequence: object) -> None:
    with pytest.raises(TraceConfigurationError, match="sequence"):
        TraceRecord.from_event(
            "trace-a", sequence, _event()  # type: ignore[arg-type]
        )


def test_record_reuses_agent_event_fields_and_preserves_utc() -> None:
    offset = timezone(timedelta(hours=8))
    event = _event(
        EventType.PLANNER_FAILED,
        occurred_at=datetime(2026, 3, 4, 13, 6, 7, tzinfo=offset),
        run_id="run-a",
        thread_id="thread-a",
        failure_category=FailureCategory.VALIDATION,
    )
    record = InMemoryTraceRecorder().record("trace-a", event)
    assert record.event_type is event.event_type
    assert record.stage is event.stage
    assert record.outcome is event.outcome
    assert record.occurred_at == FIXED_TIME
    assert record.failure_category is FailureCategory.VALIDATION


def test_sequence_is_strict_and_same_timestamp_does_not_affect_order() -> None:
    recorder = InMemoryTraceRecorder()
    first = recorder.record("trace-a", _event())
    second = recorder.record(
        "trace-a", _event(EventType.PLANNER_COMPLETED)
    )
    trace = recorder.get_trace("trace-a")
    assert first.sequence == 1 and second.sequence == 2
    assert tuple(item.sequence for item in trace.records) == (1, 2)
    assert trace.records[0].occurred_at == trace.records[1].occurred_at
    assert trace.records[0].event_type is EventType.PLANNER_STARTED


def test_trace_ids_are_isolated_and_empty_trace_is_stable() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record("trace-a", _event())
    recorder.record("trace-b", _event(EventType.PLANNER_COMPLETED))
    assert len(recorder.get_trace("trace-a").records) == 1
    assert recorder.get_trace("trace-a").records[0].sequence == 1
    assert recorder.get_trace("trace-b").records[0].sequence == 1
    assert recorder.get_trace("trace-empty") == ExecutionTrace(
        "trace-empty", ()
    )


def test_run_and_thread_identity_are_independent_and_never_invented() -> None:
    recorder = InMemoryTraceRecorder()
    missing = recorder.record("trace-a", _event())
    present = recorder.record(
        "trace-a", _event(run_id="run-1", thread_id="thread-2")
    )
    assert missing.run_id is None and missing.thread_id is None
    assert present.run_id == "run-1"
    assert present.thread_id == "thread-2"


def test_trace_contract_has_no_arbitrary_sensitive_payload() -> None:
    names = {field.name for field in fields(TraceRecord)}
    assert "payload" not in names
    assert "metadata" not in names
    assert "prompt" not in names
    assert "arguments" not in names
    assert "exception" not in names


def test_snapshot_cannot_be_changed_through_recorder_or_external_collection() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record("trace-a", _event())
    snapshot = recorder.get_trace("trace-a")
    recorder.record("trace-a", _event(EventType.PLANNER_COMPLETED))
    assert len(snapshot.records) == 1
    assert len(recorder.get_trace("trace-a").records) == 2
    with pytest.raises(AttributeError):
        snapshot.records.append(_record(sequence=2))  # type: ignore[attr-defined]


def test_capacity_evicts_oldest_record_deterministically() -> None:
    recorder = InMemoryTraceRecorder(capacity=2)
    recorder.record("trace-a", _event())
    recorder.record("trace-a", _event(EventType.PLANNER_COMPLETED))
    recorder.record("trace-b", _event())
    assert tuple(
        item.sequence for item in recorder.get_trace("trace-a").records
    ) == (2,)
    assert tuple(
        item.sequence for item in recorder.get_trace("trace-b").records
    ) == (1,)
    with pytest.raises(TraceConfigurationError, match="capacity"):
        InMemoryTraceRecorder(capacity=0)


def test_capacity_bounds_records_and_trace_indexes_for_many_trace_ids() -> None:
    recorder = InMemoryTraceRecorder(capacity=2)
    for index in range(100):
        recorder.record(f"trace-{index}", _event())
    assert len(recorder._order) == 2
    assert len(recorder._records) == 2
    assert len(recorder._next_sequence) == 2
    assert recorder.get_trace("trace-0").records == ()


def test_partially_evicted_trace_keeps_monotonic_sequence() -> None:
    recorder = InMemoryTraceRecorder(capacity=2)
    recorder.record("trace-a", _event())
    recorder.record("trace-a", _event(EventType.PLANNER_COMPLETED))
    recorder.record("trace-b", _event())
    appended = recorder.record("trace-a", _event(EventType.PLANNER_FAILED))
    assert appended.sequence == 3
    assert tuple(
        item.sequence for item in recorder.get_trace("trace-a").records
    ) == (3,)


def test_fully_evicted_trace_starts_a_new_retention_cycle() -> None:
    recorder = InMemoryTraceRecorder(capacity=1)
    recorder.record("trace-a", _event())
    recorder.record("trace-b", _event())
    assert recorder.get_trace("trace-a").records == ()
    restarted = recorder.record("trace-a", _event())
    assert restarted.sequence == 1
    assert tuple(
        item.sequence for item in recorder.get_trace("trace-a").records
    ) == (1,)


def test_concurrent_writes_to_same_trace_have_unique_ordered_sequences() -> None:
    recorder = InMemoryTraceRecorder(capacity=100)
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(
            executor.map(
                lambda _: recorder.record("trace-a", _event()),
                range(50),
            )
        )
    assert {record.sequence for record in records} == set(range(1, 51))
    assert tuple(
        record.sequence for record in recorder.get_trace("trace-a").records
    ) == tuple(range(1, 51))


def test_snapshot_is_unchanged_after_its_records_are_evicted() -> None:
    recorder = InMemoryTraceRecorder(capacity=1)
    recorder.record("trace-a", _event())
    snapshot = recorder.get_trace("trace-a")
    recorder.record("trace-b", _event())
    assert recorder.get_trace("trace-a").records == ()
    assert len(snapshot.records) == 1
    assert snapshot.records[0].trace_id == "trace-a"


def test_recorder_only_consumes_agent_events_and_has_no_external_dependencies() -> None:
    recorder = InMemoryTraceRecorder()
    with pytest.raises(TraceConfigurationError, match="AgentEvent"):
        recorder.record("trace-a", object())  # type: ignore[arg-type]
    assert not hasattr(recorder, "client")
    assert not hasattr(recorder, "llm_service")
