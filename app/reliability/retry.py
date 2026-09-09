import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
import math
from typing import TypeVar

from app.reliability.failures import (
    FailureCategory,
    FailureDescriptor,
    classify_failure,
)


class OperationSemantics(StrEnum):
    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 2
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be at least 1")
        if (
            not isinstance(self.delay_seconds, (int, float))
            or isinstance(self.delay_seconds, bool)
            or not math.isfinite(self.delay_seconds)
            or self.delay_seconds < 0
        ):
            raise ValueError("delay_seconds must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class RetryDecision:
    should_retry: bool
    reason: str


def decide_retry(
    failure: FailureDescriptor,
    operation: OperationSemantics,
    *,
    attempt: int,
    policy: RetryPolicy,
) -> RetryDecision:
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("attempt must be a positive 1-based integer")
    if failure.category is FailureCategory.CANCELLED:
        return RetryDecision(False, "cancelled")
    if failure.category is FailureCategory.BUSINESS_TERMINAL:
        return RetryDecision(False, "business_terminal")
    if failure.category is FailureCategory.AMBIGUOUS_SIDE_EFFECT:
        return RetryDecision(False, "ambiguous_side_effect")
    if attempt >= policy.max_attempts:
        return RetryDecision(False, "attempt_limit_reached")
    if operation is OperationSemantics.SIDE_EFFECTING:
        return RetryDecision(False, "side_effecting_operation")
    if operation is not OperationSemantics.READ_ONLY:
        return RetryDecision(False, "unknown_operation_semantics")
    if failure.category is not FailureCategory.TRANSIENT:
        return RetryDecision(False, "non_transient_failure")
    return RetryDecision(True, "transient_read_retry")


T = TypeVar("T")


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    semantics: OperationSemantics,
    policy: RetryPolicy = RetryPolicy(),
    classifier: Callable[[Exception], FailureDescriptor] = classify_failure,
) -> T:
    attempt = 1
    while True:
        try:
            return await operation()
        except Exception as exc:
            decision = decide_retry(
                classifier(exc),
                semantics,
                attempt=attempt,
                policy=policy,
            )
            if not decision.should_retry:
                raise
            if policy.delay_seconds:
                await asyncio.sleep(policy.delay_seconds)
            attempt += 1
