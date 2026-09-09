from dataclasses import dataclass

from app.reliability.failures import (
    FailureCategory,
    FailureDescriptor,
    FailureDomain,
)
from app.reliability.retry import OperationSemantics


@dataclass(frozen=True, slots=True)
class DegradationDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class DegradationRecord:
    component: str
    domain: FailureDomain
    category: FailureCategory
    code: str
    safe_message: str

    @classmethod
    def from_failure(
        cls,
        component: str,
        failure: FailureDescriptor,
    ) -> "DegradationRecord":
        return cls(
            component=component,
            domain=failure.domain,
            category=failure.category,
            code=failure.code,
            safe_message=failure.safe_message,
        )


def decide_degradation(
    failure: FailureDescriptor,
    operation: OperationSemantics,
    *,
    optional: bool,
) -> DegradationDecision:
    if failure.category is FailureCategory.CANCELLED:
        return DegradationDecision(False, "cancelled")
    if failure.category is FailureCategory.AMBIGUOUS_SIDE_EFFECT:
        return DegradationDecision(False, "ambiguous_side_effect")
    if failure.category is FailureCategory.BUSINESS_TERMINAL:
        return DegradationDecision(False, "business_terminal")
    if not optional:
        return DegradationDecision(False, "required_component")
    if operation is OperationSemantics.SIDE_EFFECTING:
        return DegradationDecision(False, "side_effecting_operation")
    if operation is not OperationSemantics.READ_ONLY:
        return DegradationDecision(False, "unknown_operation_semantics")
    if failure.category is not FailureCategory.TRANSIENT:
        return DegradationDecision(False, "non_transient_failure")
    return DegradationDecision(True, "eligible_transient_optional_read")
