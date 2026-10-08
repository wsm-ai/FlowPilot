from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import math

from app.evaluation.models import EvaluationResult, EvaluationStatus


class EvaluationMetricConfigurationError(Exception):
    """Raised when evaluation results cannot be aggregated safely."""


class EvaluationDimension(StrEnum):
    PLANNER = "planner"
    HITL = "hitl"
    RAG = "rag"
    MCP = "mcp"
    RELIABILITY = "reliability"


_DIMENSION_ORDER = tuple(EvaluationDimension)
_DIMENSIONS_BY_NAMESPACE = {
    dimension.value: dimension for dimension in _DIMENSION_ORDER
}


def _validate_count(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvaluationMetricConfigurationError(
            f"{name} must be a non-negative integer"
        )


def _validate_rate(value: float | None, name: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise EvaluationMetricConfigurationError(
            f"{name} must be between zero and one"
        )


@dataclass(frozen=True, slots=True)
class DimensionMetrics:
    dimension: EvaluationDimension
    case_total: int
    case_passed: int
    case_failed: int
    case_errors: int
    behavior_checks_total: int
    behavior_checks_passed: int
    behavior_checks_failed: int
    case_pass_rate: float
    evaluation_error_rate: float
    behavior_check_pass_rate: float | None

    def __post_init__(self) -> None:
        for name in (
            "case_total",
            "case_passed",
            "case_failed",
            "case_errors",
            "behavior_checks_total",
            "behavior_checks_passed",
            "behavior_checks_failed",
        ):
            _validate_count(getattr(self, name), name)
        for name in (
            "case_pass_rate",
            "evaluation_error_rate",
            "behavior_check_pass_rate",
        ):
            _validate_rate(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    case_total: int
    case_passed: int
    case_failed: int
    case_errors: int
    behavior_checks_total: int
    behavior_checks_passed: int
    behavior_checks_failed: int
    case_pass_rate: float
    evaluation_error_rate: float
    behavior_check_pass_rate: float | None
    dimensions: tuple[DimensionMetrics, ...]

    def __post_init__(self) -> None:
        for name in (
            "case_total",
            "case_passed",
            "case_failed",
            "case_errors",
            "behavior_checks_total",
            "behavior_checks_passed",
            "behavior_checks_failed",
        ):
            _validate_count(getattr(self, name), name)
        for name in (
            "case_pass_rate",
            "evaluation_error_rate",
            "behavior_check_pass_rate",
        ):
            _validate_rate(getattr(self, name), name)
        if not isinstance(self.dimensions, tuple):
            raise EvaluationMetricConfigurationError(
                "dimensions must be an immutable tuple"
            )


@dataclass(slots=True)
class _Counts:
    case_total: int = 0
    case_passed: int = 0
    case_failed: int = 0
    case_errors: int = 0
    behavior_checks_total: int = 0
    behavior_checks_passed: int = 0
    behavior_checks_failed: int = 0

    def include(self, result: EvaluationResult) -> None:
        self.case_total += 1
        if result.status is EvaluationStatus.PASSED:
            self.case_passed += 1
        elif result.status is EvaluationStatus.FAILED:
            self.case_failed += 1
        else:
            self.case_errors += 1

        if result.status is EvaluationStatus.ERROR:
            return
        passed = sum(check.passed for check in result.checks)
        self.behavior_checks_total += len(result.checks)
        self.behavior_checks_passed += passed
        self.behavior_checks_failed += len(result.checks) - passed


def _dimension_for(case_id: str) -> EvaluationDimension:
    namespace, separator, identifier = case_id.partition(".")
    if (
        not separator
        or not identifier.strip()
        or namespace not in _DIMENSIONS_BY_NAMESPACE
    ):
        raise EvaluationMetricConfigurationError(
            "Evaluation result has an unknown case namespace"
        )
    return _DIMENSIONS_BY_NAMESPACE[namespace]


def _rates(counts: _Counts) -> tuple[float, float, float | None]:
    if counts.case_total:
        case_pass_rate = counts.case_passed / counts.case_total
        error_rate = counts.case_errors / counts.case_total
    else:
        case_pass_rate = 0.0
        error_rate = 0.0
    behavior_rate = (
        counts.behavior_checks_passed / counts.behavior_checks_total
        if counts.behavior_checks_total
        else None
    )
    return case_pass_rate, error_rate, behavior_rate


def _dimension_metrics(
    dimension: EvaluationDimension,
    counts: _Counts,
) -> DimensionMetrics:
    case_rate, error_rate, behavior_rate = _rates(counts)
    return DimensionMetrics(
        dimension=dimension,
        case_total=counts.case_total,
        case_passed=counts.case_passed,
        case_failed=counts.case_failed,
        case_errors=counts.case_errors,
        behavior_checks_total=counts.behavior_checks_total,
        behavior_checks_passed=counts.behavior_checks_passed,
        behavior_checks_failed=counts.behavior_checks_failed,
        case_pass_rate=case_rate,
        evaluation_error_rate=error_rate,
        behavior_check_pass_rate=behavior_rate,
    )


def calculate_metrics(
    results: Sequence[EvaluationResult],
) -> EvaluationMetrics:
    """Aggregate an existing evaluation run without executing any scenarios."""
    seen_case_ids: set[str] = set()
    overall = _Counts()
    per_dimension = {dimension: _Counts() for dimension in _DIMENSION_ORDER}

    for result in results:
        if not isinstance(result.status, EvaluationStatus):
            raise EvaluationMetricConfigurationError(
                "Evaluation result has an invalid status"
            )
        if result.case_id in seen_case_ids:
            raise EvaluationMetricConfigurationError(
                "Duplicate evaluation result case_id"
            )
        seen_case_ids.add(result.case_id)
        dimension = _dimension_for(result.case_id)
        overall.include(result)
        per_dimension[dimension].include(result)

    case_rate, error_rate, behavior_rate = _rates(overall)
    return EvaluationMetrics(
        case_total=overall.case_total,
        case_passed=overall.case_passed,
        case_failed=overall.case_failed,
        case_errors=overall.case_errors,
        behavior_checks_total=overall.behavior_checks_total,
        behavior_checks_passed=overall.behavior_checks_passed,
        behavior_checks_failed=overall.behavior_checks_failed,
        case_pass_rate=case_rate,
        evaluation_error_rate=error_rate,
        behavior_check_pass_rate=behavior_rate,
        dimensions=tuple(
            _dimension_metrics(dimension, per_dimension[dimension])
            for dimension in _DIMENSION_ORDER
        ),
    )
