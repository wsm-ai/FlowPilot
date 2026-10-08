from dataclasses import dataclass
from enum import StrEnum
import re

from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.models import EvaluationStatus


REPORT_SCHEMA_VERSION = "1.0"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_SUITE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,127}$")


class EvaluationReportConfigurationError(ValueError):
    """Raised when a deterministic report cannot be built safely."""


class ReportFailureCode(StrEnum):
    CHECK_FAILED = "check_failed"
    SCENARIO_ERROR = "scenario_error"


def validate_report_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise EvaluationReportConfigurationError(
            f"{field_name} is not a safe report identifier"
        )
    return value


@dataclass(frozen=True, slots=True)
class EvaluationReportCheck:
    name: str
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", validate_report_identifier(self.name, "check name")
        )
        if not isinstance(self.passed, bool):
            raise EvaluationReportConfigurationError(
                "report check passed must be a bool"
            )


@dataclass(frozen=True, slots=True)
class EvaluationScenarioReport:
    case_id: str
    status: EvaluationStatus
    checks: tuple[EvaluationReportCheck, ...]
    failure_code: ReportFailureCode | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "case_id", validate_report_identifier(self.case_id, "case_id")
        )
        if not isinstance(self.status, EvaluationStatus):
            raise EvaluationReportConfigurationError(
                "report scenario status is invalid"
            )
        if not isinstance(self.checks, tuple):
            raise EvaluationReportConfigurationError(
                "report scenario checks must be an immutable tuple"
            )
        if not all(isinstance(check, EvaluationReportCheck) for check in self.checks):
            raise EvaluationReportConfigurationError(
                "report scenario contains an invalid check"
            )
        if self.failure_code is not None and not isinstance(
            self.failure_code, ReportFailureCode
        ):
            raise EvaluationReportConfigurationError(
                "report failure code is invalid"
            )
        expected_failure_code = {
            EvaluationStatus.PASSED: None,
            EvaluationStatus.FAILED: ReportFailureCode.CHECK_FAILED,
            EvaluationStatus.ERROR: ReportFailureCode.SCENARIO_ERROR,
        }[self.status]
        if self.failure_code is not expected_failure_code:
            raise EvaluationReportConfigurationError(
                "report scenario failure code is inconsistent"
            )


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: str
    suite_name: str
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    error_scenarios: int
    metrics: EvaluationMetrics
    scenarios: tuple[EvaluationScenarioReport, ...]

    def __post_init__(self) -> None:
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise EvaluationReportConfigurationError(
                "unsupported evaluation report schema version"
            )
        if (
            not isinstance(self.suite_name, str)
            or _SAFE_SUITE_NAME.fullmatch(self.suite_name) is None
        ):
            raise EvaluationReportConfigurationError(
                "suite_name contains unsupported characters"
            )
        counts = (
            self.total_scenarios,
            self.passed_scenarios,
            self.failed_scenarios,
            self.error_scenarios,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise EvaluationReportConfigurationError(
                "report scenario counts must be non-negative integers"
            )
        if not isinstance(self.metrics, EvaluationMetrics):
            raise EvaluationReportConfigurationError("report metrics are invalid")
        if not isinstance(self.scenarios, tuple):
            raise EvaluationReportConfigurationError(
                "report scenarios must be an immutable tuple"
            )
        if not all(
            isinstance(scenario, EvaluationScenarioReport)
            for scenario in self.scenarios
        ):
            raise EvaluationReportConfigurationError(
                "report contains an invalid scenario"
            )
        case_ids = tuple(scenario.case_id for scenario in self.scenarios)
        if len(case_ids) != len(set(case_ids)):
            raise EvaluationReportConfigurationError(
                "report contains duplicate scenario identifiers"
            )
        if self.total_scenarios != len(self.scenarios):
            raise EvaluationReportConfigurationError(
                "report scenario total is inconsistent"
            )
        if self.total_scenarios != (
            self.passed_scenarios
            + self.failed_scenarios
            + self.error_scenarios
        ):
            raise EvaluationReportConfigurationError(
                "report scenario status counts are inconsistent"
            )
        actual_passed = sum(
            scenario.status is EvaluationStatus.PASSED
            for scenario in self.scenarios
        )
        actual_failed = sum(
            scenario.status is EvaluationStatus.FAILED
            for scenario in self.scenarios
        )
        actual_errors = sum(
            scenario.status is EvaluationStatus.ERROR
            for scenario in self.scenarios
        )
        if (
            self.passed_scenarios != actual_passed
            or self.failed_scenarios != actual_failed
            or self.error_scenarios != actual_errors
        ):
            raise EvaluationReportConfigurationError(
                "report scenario status counts are inconsistent"
            )
        if (
            self.metrics.case_total != self.total_scenarios
            or self.metrics.case_passed != self.passed_scenarios
            or self.metrics.case_failed != self.failed_scenarios
            or self.metrics.case_errors != self.error_scenarios
        ):
            raise EvaluationReportConfigurationError(
                "report metrics are inconsistent with scenarios"
            )
