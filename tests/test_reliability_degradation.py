from dataclasses import FrozenInstanceError

import pytest

from app.reliability.degradation import (
    DegradationRecord,
    decide_degradation,
)
from app.reliability.failures import (
    FailureCategory,
    FailureDescriptor,
    FailureDomain,
)
from app.reliability.retry import OperationSemantics


def failure(category):
    return FailureDescriptor(
        domain=FailureDomain.MCP,
        category=category,
        code="safe_code",
        safe_message="Safe message",
    )


@pytest.mark.parametrize(
    ("category", "semantics", "optional", "allowed", "reason"),
    [
        (FailureCategory.TRANSIENT, OperationSemantics.READ_ONLY, True,
         True, "eligible_transient_optional_read"),
        (FailureCategory.TRANSIENT, OperationSemantics.READ_ONLY, False,
         False, "required_component"),
        (FailureCategory.PERMANENT, OperationSemantics.READ_ONLY, True,
         False, "non_transient_failure"),
        (FailureCategory.CONFIGURATION, OperationSemantics.READ_ONLY, True,
         False, "non_transient_failure"),
        (FailureCategory.VALIDATION, OperationSemantics.READ_ONLY, True,
         False, "non_transient_failure"),
        (FailureCategory.AMBIGUOUS_SIDE_EFFECT,
         OperationSemantics.READ_ONLY, True, False, "ambiguous_side_effect"),
        (FailureCategory.CANCELLED, OperationSemantics.READ_ONLY, True,
         False, "cancelled"),
        (FailureCategory.BUSINESS_TERMINAL, OperationSemantics.READ_ONLY, True,
         False, "business_terminal"),
        (FailureCategory.TRANSIENT, OperationSemantics.SIDE_EFFECTING, True,
         False, "side_effecting_operation"),
        (FailureCategory.TRANSIENT, OperationSemantics.UNKNOWN, True,
         False, "unknown_operation_semantics"),
    ],
)
def test_degradation_policy_is_deterministic(
    category, semantics, optional, allowed, reason
):
    decision = decide_degradation(
        failure(category), semantics, optional=optional
    )
    assert (decision.allowed, decision.reason) == (allowed, reason)


def test_degradation_record_is_safe_and_immutable():
    descriptor = failure(FailureCategory.TRANSIENT)
    record = DegradationRecord.from_failure("github-readonly", descriptor)

    assert record.component == "github-readonly"
    assert record.safe_message == "Safe message"
    assert "TOKEN=secret" not in repr(record)
    with pytest.raises(FrozenInstanceError):
        record.code = "changed"
