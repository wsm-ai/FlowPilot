from typing import Any, Protocol

from app.persistence.models import AgentRunRecord, RunStatus


class PersistenceError(Exception):
    """Raised when agent run persistence cannot complete an operation."""


class RunRepository(Protocol):
    async def initialize(self) -> None:
        ...

    async def create(self, record: AgentRunRecord) -> None:
        ...

    async def get(self, run_id: str) -> AgentRunRecord | None:
        ...

    async def update(
        self,
        run_id: str,
        *,
        status: RunStatus,
        result: dict[str, Any] | None = None,
        error_type: str | None = None,
    ) -> AgentRunRecord:
        ...

    async def list_recent(self, limit: int = 20) -> list[AgentRunRecord]:
        ...
