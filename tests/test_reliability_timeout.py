import asyncio
from dataclasses import FrozenInstanceError

import pytest

from app.reliability.failures import FailureCategory, FailureDomain, classify_failure
from app.reliability.retry import OperationSemantics, RetryPolicy, decide_retry
from app.reliability.timeouts import (
    ReadOperationTimeoutError,
    SideEffectOperationTimeoutError,
    TimeoutPolicy,
    UnknownOperationTimeoutError,
    run_with_timeout,
)


def test_operation_completes_before_deadline_and_is_called_once():
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        return {"ok": True}

    result = asyncio.run(
        run_with_timeout(
            operation,
            semantics=OperationSemantics.READ_ONLY,
            policy=TimeoutPolicy(1),
        )
    )

    assert result == {"ok": True}
    assert calls == 1


@pytest.mark.parametrize(
    ("semantics", "error_type", "safe_message"),
    [
        (OperationSemantics.READ_ONLY, ReadOperationTimeoutError,
         "Read operation timed out"),
        (OperationSemantics.SIDE_EFFECTING, SideEffectOperationTimeoutError,
         "Side-effect operation timed out"),
        (OperationSemantics.UNKNOWN, UnknownOperationTimeoutError,
         "Operation timed out"),
    ],
)
def test_deadline_maps_to_safe_semantics_specific_error(
    semantics, error_type, safe_message
):
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)

    with pytest.raises(error_type, match=f"^{safe_message}$") as raised:
        asyncio.run(
            run_with_timeout(
                operation,
                semantics=semantics,
                policy=TimeoutPolicy(0.01),
            )
        )

    assert calls == 1
    assert "secret" not in str(raised.value)


def test_external_cancellation_propagates_unchanged():
    async def scenario():
        started = asyncio.Event()

        async def operation():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(
            run_with_timeout(
                operation,
                semantics=OperationSemantics.READ_ONLY,
                policy=TimeoutPolicy(10),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_operation_timeout_error_is_not_mistaken_for_local_deadline():
    error = TimeoutError("transport timeout detail")

    async def operation():
        raise error

    with pytest.raises(TimeoutError) as raised:
        asyncio.run(
            run_with_timeout(
                operation,
                semantics=OperationSemantics.READ_ONLY,
                policy=TimeoutPolicy(1),
            )
        )
    assert raised.value is error


@pytest.mark.parametrize("base_error", [KeyboardInterrupt(), SystemExit(9)])
def test_process_control_base_exceptions_are_not_swallowed(base_error):
    async def operation():
        raise base_error

    with pytest.raises(type(base_error)) as raised:
        asyncio.run(
            run_with_timeout(
                operation,
                semantics=OperationSemantics.UNKNOWN,
                policy=TimeoutPolicy(1),
            )
        )
    assert raised.value is base_error


@pytest.mark.parametrize(
    "value", [0, -1, float("nan"), float("inf"), float("-inf"), True]
)
def test_timeout_policy_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive and finite"):
        TimeoutPolicy(value)


def test_timeout_policy_is_immutable():
    policy = TimeoutPolicy(1)
    with pytest.raises(FrozenInstanceError):
        policy.timeout_seconds = 2


def test_timeout_failure_classification_and_retry_decisions():
    policy = RetryPolicy(max_attempts=2)
    cases = [
        (
            ReadOperationTimeoutError("private"),
            FailureCategory.TRANSIENT,
            OperationSemantics.READ_ONLY,
            True,
        ),
        (
            SideEffectOperationTimeoutError("private"),
            FailureCategory.AMBIGUOUS_SIDE_EFFECT,
            OperationSemantics.SIDE_EFFECTING,
            False,
        ),
        (
            UnknownOperationTimeoutError("private"),
            FailureCategory.AMBIGUOUS_SIDE_EFFECT,
            OperationSemantics.UNKNOWN,
            False,
        ),
    ]

    for error, category, semantics, expected_retry in cases:
        descriptor = classify_failure(error)
        assert descriptor.domain is FailureDomain.WORKFLOW
        assert descriptor.category is category
        assert "private" not in descriptor.safe_message
        assert decide_retry(
            descriptor, semantics, attempt=1, policy=policy
        ).should_retry is expected_retry
