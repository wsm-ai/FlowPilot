from dataclasses import dataclass
from enum import StrEnum
import math

from app.evaluation.metrics import EvaluationDimension


class RegressionGateConfigurationError(ValueError):
    """Raised when quality gate policy is invalid."""


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"


class GateReason(StrEnum):
    EMPTY_EVALUATION = "empty_evaluation"
    FAILED_SCENARIOS = "failed_scenarios"
    EVALUATION_ERRORS = "evaluation_errors"
    BELOW_THRESHOLD = "below_threshold"
    REGRESSION_DETECTED = "regression_detected"
    MISSING_REQUIRED_METRIC = "missing_required_metric"
    MISSING_REQUIRED_DIMENSION = "missing_required_dimension"
    INCOMPATIBLE_BASELINE = "incompatible_baseline"
    INVALID_REPORT = "invalid_report"


@dataclass(frozen=True, slots=True)
class DimensionThreshold:
    dimension: EvaluationDimension
    min_case_pass_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, EvaluationDimension):
            raise RegressionGateConfigurationError("Invalid gate dimension")
        _rate(self.min_case_pass_rate, "dimension pass rate")


@dataclass(frozen=True, slots=True)
class RegressionGateConfig:
    max_failed_scenarios: int = 0
    max_error_scenarios: int = 0
    min_case_pass_rate: float = 1.0
    min_behavior_check_pass_rate: float = 1.0
    max_regression: float = 0.0
    dimension_thresholds: tuple[DimensionThreshold, ...] = ()

    def __post_init__(self) -> None:
        _count(self.max_failed_scenarios, "max failed scenarios")
        _count(self.max_error_scenarios, "max error scenarios")
        _rate(self.min_case_pass_rate, "minimum case pass rate")
        _rate(
            self.min_behavior_check_pass_rate,
            "minimum behavior check pass rate",
        )
        _rate(self.max_regression, "maximum regression")
        if not isinstance(self.dimension_thresholds, tuple):
            raise RegressionGateConfigurationError(
                "Dimension thresholds must be an immutable tuple"
            )
        if not all(
            isinstance(item, DimensionThreshold)
            for item in self.dimension_thresholds
        ):
            raise RegressionGateConfigurationError("Invalid dimension threshold")
        dimensions = tuple(item.dimension for item in self.dimension_thresholds)
        if len(dimensions) != len(set(dimensions)):
            raise RegressionGateConfigurationError(
                "Duplicate dimension threshold"
            )


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    passed: bool
    reason: GateReason | None = None


@dataclass(frozen=True, slots=True)
class RegressionGateResult:
    gate_status: GateStatus
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    error_scenarios: int
    baseline_compared: bool
    checks: tuple[GateCheck, ...]
    reasons: tuple[GateReason, ...]


def _count(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RegressionGateConfigurationError(
            f"{name} must be a non-negative integer"
        )


def _rate(value: float, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise RegressionGateConfigurationError(
            f"{name} must be between zero and one"
        )
