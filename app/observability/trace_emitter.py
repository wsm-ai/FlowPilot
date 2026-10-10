import asyncio
import logging
from typing import Protocol

from app.observability.models import AgentEvent
from app.observability.trace import TraceRecord, TraceThreadAssociationConflictError


class TraceRepository(Protocol):
    async def append_event(
        self, trace_id: str, event: AgentEvent
    ) -> TraceRecord:
        ...

    async def associate_thread(self, thread_id: str, trace_id: str) -> None:
        ...

    async def release_thread(self, thread_id: str, trace_id: str) -> None:
        ...

    async def get_trace_id_for_thread(self, thread_id: str) -> str | None:
        ...


class AsyncTraceEmitter(Protocol):
    async def emit(self, trace_id: str, event: AgentEvent) -> None:
        ...

    async def associate_thread(self, thread_id: str, trace_id: str) -> bool:
        ...

    async def release_thread(self, thread_id: str, trace_id: str) -> None:
        ...

    async def trace_id_for_thread(self, thread_id: str) -> str | None:
        ...


class SQLiteTraceEmitter:
    """Best-effort async delivery to durable trace persistence."""

    def __init__(
        self,
        repository: TraceRepository,
        *,
        diagnostic_logger: logging.Logger | None = None,
    ) -> None:
        self._repository = repository
        self._logger = diagnostic_logger or logging.getLogger(
            "flowpilot.diagnostics"
        )

    async def emit(self, trace_id: str, event: AgentEvent) -> None:
        try:
            await self._repository.append_event(trace_id, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._diagnose("Trace event persistence failed")

    async def associate_thread(self, thread_id: str, trace_id: str) -> bool:
        try:
            await self._repository.associate_thread(thread_id, trace_id)
            return True
        except asyncio.CancelledError:
            raise
        except TraceThreadAssociationConflictError:
            raise
        except Exception:
            self._diagnose("Trace thread association failed")
            return False

    async def release_thread(self, thread_id: str, trace_id: str) -> None:
        try:
            await self._repository.release_thread(thread_id, trace_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._diagnose("Trace thread association release failed")

    async def trace_id_for_thread(self, thread_id: str) -> str | None:
        try:
            return await self._repository.get_trace_id_for_thread(thread_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._diagnose("Trace thread association lookup failed")
            return None

    def _diagnose(self, message: str) -> None:
        try:
            self._logger.warning(message)
        except Exception:
            pass


class NoOpAsyncTraceEmitter:
    """Safe degradation used when durable Trace storage is unavailable."""

    async def emit(self, trace_id: str, event: AgentEvent) -> None:
        return None

    async def associate_thread(self, thread_id: str, trace_id: str) -> bool:
        return False

    async def release_thread(self, thread_id: str, trace_id: str) -> None:
        return None

    async def trace_id_for_thread(self, thread_id: str) -> str | None:
        return None
