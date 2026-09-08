import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any, Protocol, TypeVar


class SideEffectExecutionStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    AMBIGUOUS = "ambiguous"


class SideEffectExecutionError(Exception):
    """Base error for local side-effect execution protection."""


class SideEffectReplayBlockedError(SideEffectExecutionError):
    """Raised when a prior dispatch makes automatic replay unsafe."""


class SideEffectConflictError(SideEffectExecutionError):
    """Raised when an execution identity is reused with other arguments."""


@dataclass(frozen=True, slots=True)
class SideEffectExecutionRecord:
    run_id: str
    step_id: int
    action: str
    arguments_digest: str
    status: SideEffectExecutionStatus
    result: Any | None = None


@dataclass(frozen=True, slots=True)
class SideEffectReservation:
    record: SideEffectExecutionRecord
    created: bool


class SideEffectExecutionRepository(Protocol):
    async def get(
        self, run_id: str, step_id: int, action: str
    ) -> SideEffectExecutionRecord | None:
        ...

    async def reserve_started(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
    ) -> SideEffectReservation:
        ...

    async def mark_completed(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
        result: Any,
    ) -> SideEffectExecutionRecord:
        ...

    async def mark_ambiguous(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
    ) -> SideEffectExecutionRecord:
        ...


def digest_arguments(arguments: dict[str, Any]) -> str:
    try:
        canonical = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SideEffectExecutionError(
            "Side-effect arguments are not JSON serializable"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


T = TypeVar("T")


class SideEffectExecutor:
    """Provides local at-most-once dispatch protection for side effects."""

    def __init__(self, repository: SideEffectExecutionRepository) -> None:
        self._repository = repository

    async def execute(
        self,
        *,
        run_id: str,
        step_id: int,
        action: str,
        arguments: dict[str, Any],
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        arguments_digest = digest_arguments(arguments)
        reservation = await self._repository.reserve_started(
            run_id=run_id,
            step_id=step_id,
            action=action,
            arguments_digest=arguments_digest,
        )
        existing = reservation.record
        if not reservation.created:
            if existing.arguments_digest != arguments_digest:
                raise SideEffectConflictError(
                    "Side-effect execution conflicts with existing record"
                )
            if existing.status is SideEffectExecutionStatus.COMPLETED:
                return deepcopy(existing.result)
            raise SideEffectReplayBlockedError(
                "Side-effect execution replay is blocked"
            )

        try:
            result = await operation()
        except asyncio.CancelledError:
            await self._mark_ambiguous_best_effort(
                run_id, step_id, action, arguments_digest
            )
            raise
        except Exception:
            await self._mark_ambiguous_best_effort(
                run_id, step_id, action, arguments_digest
            )
            raise

        completed = await self._repository.mark_completed(
            run_id=run_id,
            step_id=step_id,
            action=action,
            arguments_digest=arguments_digest,
            result=result,
        )
        return deepcopy(completed.result)

    async def _mark_ambiguous_best_effort(
        self,
        run_id: str,
        step_id: int,
        action: str,
        arguments_digest: str,
    ) -> None:
        try:
            await self._repository.mark_ambiguous(
                run_id=run_id,
                step_id=step_id,
                action=action,
                arguments_digest=arguments_digest,
            )
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
