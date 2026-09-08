import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from app.reliability.retry import OperationSemantics


DEFAULT_OPERATION_TIMEOUT_SECONDS = 30.0


class OperationTimeoutError(Exception):
    """Base error for an application-level operation deadline."""


class ReadOperationTimeoutError(OperationTimeoutError):
    """Raised when a read-only operation exceeds its deadline."""


class SideEffectOperationTimeoutError(OperationTimeoutError):
    """Raised when a side-effecting operation has an ambiguous timeout."""


class UnknownOperationTimeoutError(OperationTimeoutError):
    """Raised when an operation with unknown semantics times out."""


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")


T = TypeVar("T")


async def run_with_timeout(
    operation: Callable[[], Awaitable[T]],
    *,
    semantics: "OperationSemantics",
    policy: TimeoutPolicy = TimeoutPolicy(),
) -> T:
    # Kept local to avoid coupling the failure taxonomy import graph to retry.
    from app.reliability.retry import OperationSemantics

    timeout_context = asyncio.timeout(policy.timeout_seconds)
    try:
        async with timeout_context:
            return await operation()
    except TimeoutError as exc:
        # Preserve a TimeoutError raised by the operation itself. Only the
        # deadline owned by this helper receives unified semantic mapping.
        if not timeout_context.expired():
            raise
        if semantics is OperationSemantics.READ_ONLY:
            raise ReadOperationTimeoutError("Read operation timed out") from exc
        if semantics is OperationSemantics.SIDE_EFFECTING:
            raise SideEffectOperationTimeoutError(
                "Side-effect operation timed out"
            ) from exc
        raise UnknownOperationTimeoutError("Operation timed out") from exc
