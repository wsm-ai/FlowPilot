from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Protocol


class EvaluationError(Exception):
    """Base error for the offline evaluation boundary."""


class EvaluationConfigurationError(EvaluationError):
    """Raised when evaluation cases are configured incorrectly."""


class EvaluationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


def _non_blank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationConfigurationError(f"{field_name} must not be blank")
    return value.strip()


@dataclass(frozen=True, slots=True)
class EvaluationCheck:
    name: str
    passed: bool
    message: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_blank(self.name, "check name"))
        if not isinstance(self.passed, bool):
            raise EvaluationConfigurationError("check passed must be a bool")


class EvaluationScenario(Protocol):
    async def run(self) -> Sequence[EvaluationCheck]:
        ...


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    name: str
    description: str
    scenario: EvaluationScenario = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _non_blank(self.case_id, "case_id"))
        object.__setattr__(self, "name", _non_blank(self.name, "case name"))


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    status: EvaluationStatus
    checks: tuple[EvaluationCheck, ...]
    duration_ms: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise EvaluationConfigurationError(
                "evaluation duration must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float


def summarize(results: Sequence[EvaluationResult]) -> EvaluationSummary:
    total = len(results)
    passed = sum(result.status is EvaluationStatus.PASSED for result in results)
    failed = sum(result.status is EvaluationStatus.FAILED for result in results)
    errors = sum(result.status is EvaluationStatus.ERROR for result in results)
    return EvaluationSummary(
        total=total,
        passed=passed,
        failed=failed,
        errors=errors,
        pass_rate=passed / total if total else 0.0,
    )
