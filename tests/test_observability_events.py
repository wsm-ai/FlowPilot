from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import logging

import pytest

from app.observability import (
    AgentEvent,
    EventOutcome,
    EventStage,
    EventType,
    ObservabilityConfigurationError,
    StructuredEventFormatter,
    StructuredLoggingEmitter,
    create_structured_logging_emitter,
    event_to_dict,
)
from app.reliability.failures import FailureCategory


FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def _event(**overrides: object) -> AgentEvent:
    values = {
        "event_type": EventType.PLANNER_COMPLETED,
        "occurred_at": FIXED_TIME,
        "stage": EventStage.PLANNER,
        "outcome": EventOutcome.SUCCEEDED,
        "run_id": None,
        "thread_id": None,
        "failure_category": None,
    }
    values.update(overrides)
    return AgentEvent(**values)  # type: ignore[arg-type]


def test_event_enums_and_model_are_typed_and_immutable() -> None:
    event = _event()
    assert event.event_type is EventType.PLANNER_COMPLETED
    assert event.stage is EventStage.PLANNER
    assert event.outcome is EventOutcome.SUCCEEDED
    with pytest.raises(FrozenInstanceError):
        event.run_id = "changed"  # type: ignore[misc]
    with pytest.raises(ObservabilityConfigurationError, match="event_type"):
        _event(event_type="planner_completed")
    with pytest.raises(ObservabilityConfigurationError, match="inconsistent"):
        _event(
            event_type=EventType.PLANNER_STARTED,
            stage=EventStage.APPROVAL,
            outcome=EventOutcome.REJECTED,
        )


def test_timestamp_is_timezone_aware_and_normalized_to_utc() -> None:
    offset = timezone(timedelta(hours=8))
    event = _event(occurred_at=datetime(2026, 1, 2, 11, 4, 5, tzinfo=offset))
    assert event.occurred_at == FIXED_TIME
    assert event_to_dict(event)["occurred_at"] == "2026-01-02T03:04:05+00:00"
    with pytest.raises(ObservabilityConfigurationError, match="timezone-aware"):
        _event(occurred_at=datetime(2026, 1, 2))


def test_json_fields_are_stable_safe_and_round_trip() -> None:
    event = _event(
        event_type=EventType.PLANNER_FAILED,
        outcome=EventOutcome.FAILED,
        run_id="run-1",
        thread_id="thread-1",
        failure_category=FailureCategory.VALIDATION,
    )
    payload = event_to_dict(event)
    assert tuple(payload) == (
        "event_type", "occurred_at", "stage", "outcome", "run_id",
        "thread_id", "failure_category",
    )
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    serialized = json.dumps(payload)
    assert "secret" not in serialized
    assert "prompt" not in serialized
    assert "arguments" not in serialized


def test_missing_identities_stay_none_and_distinct_ids_are_preserved() -> None:
    missing = event_to_dict(_event())
    present = event_to_dict(_event(run_id="run-1", thread_id="thread-2"))
    assert missing["run_id"] is None and missing["thread_id"] is None
    assert present["run_id"] == "run-1"
    assert present["thread_id"] == "thread-2"


def test_structured_logging_handler_outputs_one_valid_json_event() -> None:
    root_handlers = tuple(logging.getLogger().handlers)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredEventFormatter())
    logger = logging.getLogger("flowpilot.tests.structured")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    StructuredLoggingEmitter(logger).emit(_event())
    payload = json.loads(stream.getvalue())
    assert payload["event_type"] == "planner_completed"
    assert tuple(logging.getLogger().handlers) == root_handlers


def test_logger_factory_is_isolated_and_does_not_duplicate_handlers() -> None:
    root_handlers = tuple(logging.getLogger().handlers)
    stream = StringIO()
    logger_name = "flowpilot.tests.factory"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    first = create_structured_logging_emitter(
        logger_name=logger_name, stream=stream
    )
    second = create_structured_logging_emitter(
        logger_name=logger_name, stream=stream
    )
    first.emit(_event())
    second.emit(_event())
    assert len(logger.handlers) == 1
    assert len(stream.getvalue().splitlines()) == 2
    assert all(json.loads(line) for line in stream.getvalue().splitlines())
    assert logger.propagate is False
    assert tuple(logging.getLogger().handlers) == root_handlers


def test_event_has_no_arbitrary_payload_or_exception_message() -> None:
    fields = AgentEvent.__dataclass_fields__
    assert "payload" not in fields and "metadata" not in fields
    event = _event(
        event_type=EventType.PLANNER_FAILED,
        outcome=EventOutcome.FAILED,
        failure_category=FailureCategory.PERMANENT,
    )
    serialized = json.dumps(event_to_dict(event))
    assert "TOKEN=private" not in serialized
