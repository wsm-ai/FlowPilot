import asyncio
from dataclasses import FrozenInstanceError

import pytest

from app.reliability.failures import (
    FailureCategory,
    FailureDescriptor,
    FailureDomain,
)
from app.reliability.retry import (
    OperationSemantics,
    RetryDecision,
    RetryPolicy,
    decide_retry,
    run_with_retry,
)


def failure(category: FailureCategory) -> FailureDescriptor:
    return FailureDescriptor(
        domain=FailureDomain.WORKFLOW,
        category=category,
        code="test_failure",
        safe_message="Test failure",
    )


@pytest.mark.parametrize(
    ("category", "operation", "attempt", "expected", "reason"),
    [
        (
            FailureCategory.TRANSIENT,
            OperationSemantics.READ_ONLY,
            1,
            True,
            "transient_read_retry",
        ),
        (
            FailureCategory.TRANSIENT,
            OperationSemantics.SIDE_EFFECTING,
            1,
            False,
            "side_effecting_operation",
        ),
        (
            FailureCategory.TRANSIENT,
            OperationSemantics.UNKNOWN,
            1,
            False,
            "unknown_operation_semantics",
        ),
        (
            FailureCategory.AMBIGUOUS_SIDE_EFFECT,
            OperationSemantics.READ_ONLY,
            1,
            False,
            "ambiguous_side_effect",
        ),
        (
            FailureCategory.CONFIGURATION,
            OperationSemantics.READ_ONLY,
            1,
            False,
            "non_transient_failure",
        ),
        (
            FailureCategory.VALIDATION,
            OperationSemantics.READ_ONLY,
            1,
            False,
            "non_transient_failure",
        ),
        (
            FailureCategory.PERMANENT,
            OperationSemantics.READ_ONLY,
            1,
            False,
            "non_transient_failure",
        ),
        (
            FailureCategory.BUSINESS_TERMINAL,
            OperationSemantics.READ_ONLY,
            1,
            False,
            "business_terminal",
        ),
        (
            FailureCategory.CANCELLED,
            OperationSemantics.READ_ONLY,
            1,
            False,
            "cancelled",
        ),
        (
            FailureCategory.TRANSIENT,
            OperationSemantics.READ_ONLY,
            2,
            False,
            "attempt_limit_reached",
        ),
    ],
)
def test_retry_decision_rules(category, operation, attempt, expected, reason):
    decision = decide_retry(
        failure(category),
        operation,
        attempt=attempt,
        policy=RetryPolicy(max_attempts=2, delay_seconds=0),
    )

    assert decision == RetryDecision(expected, reason)


def test_ambiguous_side_effect_overrides_attempt_and_operation_metadata():
    decision = decide_retry(
        failure(FailureCategory.AMBIGUOUS_SIDE_EFFECT),
        OperationSemantics.READ_ONLY,
        attempt=2,
        policy=RetryPolicy(max_attempts=2, delay_seconds=0),
    )

    assert decision == RetryDecision(False, "ambiguous_side_effect")


def test_unrecognized_operation_value_fails_closed():
    decision = decide_retry(
        failure(FailureCategory.TRANSIENT),
        "read_only",
        attempt=1,
        policy=RetryPolicy(max_attempts=2, delay_seconds=0),
    )

    assert decision == RetryDecision(False, "unknown_operation_semantics")


def test_runner_retries_transient_read_once_then_returns_success():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary")
        return "ok"

    result = asyncio.run(
        run_with_retry(
            operation,
            semantics=OperationSemantics.READ_ONLY,
            policy=RetryPolicy(delay_seconds=0),
            classifier=lambda exc: failure(FailureCategory.TRANSIENT),
        )
    )

    assert result == "ok"
    assert calls == 2


def test_runner_reraises_last_exception_at_attempt_limit():
    errors = [OSError("first"), OSError("second")]
    calls = 0

    async def operation():
        nonlocal calls
        error = errors[calls]
        calls += 1
        raise error

    with pytest.raises(OSError) as raised:
        asyncio.run(
            run_with_retry(
                operation,
                semantics=OperationSemantics.READ_ONLY,
                policy=RetryPolicy(delay_seconds=0),
                classifier=lambda exc: failure(FailureCategory.TRANSIENT),
            )
        )

    assert raised.value is errors[1]
    assert calls == 2


@pytest.mark.parametrize(
    ("semantics", "category"),
    [
        (OperationSemantics.SIDE_EFFECTING, FailureCategory.TRANSIENT),
        (OperationSemantics.UNKNOWN, FailureCategory.TRANSIENT),
        (OperationSemantics.READ_ONLY, FailureCategory.PERMANENT),
    ],
)
def test_runner_does_not_retry_unsafe_or_permanent_operations(
    semantics, category
):
    error = RuntimeError("private")
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            run_with_retry(
                operation,
                semantics=semantics,
                policy=RetryPolicy(delay_seconds=0),
                classifier=lambda exc: failure(category),
            )
        )

    assert raised.value is error
    assert error.args == ("private",)
    assert calls == 1


def test_runner_successful_first_attempt_runs_once():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        return 42

    result = asyncio.run(
        run_with_retry(
            operation,
            semantics=OperationSemantics.READ_ONLY,
            policy=RetryPolicy(delay_seconds=0),
        )
    )

    assert result == 42
    assert calls == 1


def test_runner_propagates_cancellation_without_retry():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_with_retry(
                operation,
                semantics=OperationSemantics.READ_ONLY,
                policy=RetryPolicy(delay_seconds=0),
            )
        )

    assert calls == 1


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_retry_policy_rejects_invalid_attempt_limit(max_attempts):
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=max_attempts)


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf"), -float("inf")])
def test_retry_policy_rejects_invalid_delay(delay):
    with pytest.raises(ValueError, match="delay_seconds"):
        RetryPolicy(delay_seconds=delay)


def test_policy_and_decision_are_immutable():
    policy = RetryPolicy()
    decision = RetryDecision(False, "test")

    with pytest.raises(FrozenInstanceError):
        policy.max_attempts = 3
    with pytest.raises(FrozenInstanceError):
        decision.should_retry = True


def test_decide_retry_requires_a_positive_one_based_attempt():
    with pytest.raises(ValueError, match="1-based"):
        decide_retry(
            failure(FailureCategory.TRANSIENT),
            OperationSemantics.READ_ONLY,
            attempt=0,
            policy=RetryPolicy(),
        )
