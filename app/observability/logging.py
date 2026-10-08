import json
import logging
from typing import TextIO

from app.observability.models import AgentEvent


_EVENT_FIELDS = (
    "event_type",
    "occurred_at",
    "stage",
    "outcome",
    "run_id",
    "thread_id",
    "failure_category",
)


def event_to_dict(event: AgentEvent) -> dict[str, str | None]:
    return {
        "event_type": event.event_type.value,
        "occurred_at": event.occurred_at.isoformat(),
        "stage": event.stage.value,
        "outcome": event.outcome.value,
        "run_id": event.run_id,
        "thread_id": event.thread_id,
        "failure_category": (
            event.failure_category.value
            if event.failure_category is not None
            else None
        ),
    }


class StructuredEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = record.msg
        if not isinstance(event, AgentEvent):
            raise TypeError("StructuredEventFormatter requires an AgentEvent")
        payload = event_to_dict(event)
        if tuple(payload) != _EVENT_FIELDS:
            raise RuntimeError("Structured event fields are invalid")
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class StructuredLoggingEmitter:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, event: AgentEvent) -> None:
        self._logger.info(event)


_HANDLER_MARKER = "_flowpilot_structured_event_handler"


def create_structured_logging_emitter(
    *,
    logger_name: str = "flowpilot.observability",
    stream: TextIO | None = None,
) -> StructuredLoggingEmitter:
    """Configure an isolated application event logger exactly once."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        getattr(handler, _HANDLER_MARKER, False)
        for handler in logger.handlers
    ):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredEventFormatter())
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    return StructuredLoggingEmitter(logger)
