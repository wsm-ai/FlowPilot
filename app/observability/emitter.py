from typing import Protocol

from app.observability.models import AgentEvent


class EventEmitter(Protocol):
    def emit(self, event: AgentEvent) -> None:
        ...


class NoOpEventEmitter:
    def emit(self, event: AgentEvent) -> None:
        return None


def emit_best_effort(emitter: EventEmitter, event: AgentEvent) -> None:
    """Deliver an auxiliary event without changing the business result."""
    try:
        emitter.emit(event)
    except Exception:
        pass
