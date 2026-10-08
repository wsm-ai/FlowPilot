import json
import math
from pathlib import Path
from typing import Any

from app.evaluation.metrics import (
    DimensionMetrics,
    EvaluationDimension,
    EvaluationMetrics,
    EvaluationMetricConfigurationError,
    calculate_metrics,
)
from app.evaluation.models import EvaluationCheck, EvaluationResult, EvaluationStatus
from app.evaluation.report_models import (
    REPORT_SCHEMA_VERSION,
    EvaluationReport,
    EvaluationReportCheck,
    EvaluationReportConfigurationError,
    EvaluationScenarioReport,
    ReportFailureCode,
)


MAX_REPORT_BYTES = 2 * 1024 * 1024


class EvaluationReportValidationError(ValueError):
    """Raised when an untrusted report is not a valid 11F artifact."""


def load_evaluation_report(path: str | Path) -> EvaluationReport:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise EvaluationReportValidationError("Evaluation report is unavailable") from exc
    if len(raw) > MAX_REPORT_BYTES:
        raise EvaluationReportValidationError("Evaluation report exceeds size limit")
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return _parse_report(payload)
    except EvaluationReportValidationError:
        raise
    except (
        EvaluationMetricConfigurationError,
        EvaluationReportConfigurationError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise EvaluationReportValidationError("Evaluation report is invalid") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationReportValidationError(
                "Evaluation report contains duplicate fields"
            )
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise EvaluationReportValidationError("Evaluation report contains invalid numbers")


def _fields(payload: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise EvaluationReportValidationError("Evaluation report structure is invalid")
    return payload


def _parse_report(payload: Any) -> EvaluationReport:
    item = _fields(payload, {
        "schema_version", "suite_name", "total_scenarios", "passed_scenarios",
        "failed_scenarios", "error_scenarios", "metrics", "scenarios",
    })
    if item["schema_version"] != REPORT_SCHEMA_VERSION:
        raise EvaluationReportValidationError("Evaluation report schema is unsupported")
    raw_scenarios = item["scenarios"]
    if not isinstance(raw_scenarios, list):
        raise EvaluationReportValidationError("Evaluation report scenarios are invalid")
    scenarios = tuple(_parse_scenario(value) for value in raw_scenarios)
    metrics = _parse_metrics(item["metrics"])
    report = EvaluationReport(
        schema_version=item["schema_version"],
        suite_name=item["suite_name"],
        total_scenarios=item["total_scenarios"],
        passed_scenarios=item["passed_scenarios"],
        failed_scenarios=item["failed_scenarios"],
        error_scenarios=item["error_scenarios"],
        metrics=metrics,
        scenarios=scenarios,
    )
    _validate_recalculated_metrics(report)
    return report


def _parse_scenario(payload: Any) -> EvaluationScenarioReport:
    item = _fields(payload, {"case_id", "status", "checks", "failure_code"})
    checks = item["checks"]
    if not isinstance(checks, list):
        raise EvaluationReportValidationError("Evaluation report checks are invalid")
    return EvaluationScenarioReport(
        case_id=item["case_id"],
        status=EvaluationStatus(item["status"]),
        checks=tuple(_parse_check(value) for value in checks),
        failure_code=(
            None
            if item["failure_code"] is None
            else ReportFailureCode(item["failure_code"])
        ),
    )


def _parse_check(payload: Any) -> EvaluationReportCheck:
    item = _fields(payload, {"name", "passed"})
    return EvaluationReportCheck(name=item["name"], passed=item["passed"])


def _parse_metrics(payload: Any) -> EvaluationMetrics:
    item = _fields(payload, {
        "case_total", "case_passed", "case_failed", "case_errors",
        "behavior_checks_total", "behavior_checks_passed",
        "behavior_checks_failed", "case_pass_rate", "evaluation_error_rate",
        "behavior_check_pass_rate", "dimensions",
    })
    raw_dimensions = item["dimensions"]
    if not isinstance(raw_dimensions, list):
        raise EvaluationReportValidationError("Evaluation report dimensions are invalid")
    dimensions = tuple(_parse_dimension(value) for value in raw_dimensions)
    names = tuple(value.dimension for value in dimensions)
    if len(names) != len(set(names)):
        raise EvaluationReportValidationError("Evaluation report dimensions are duplicated")
    return EvaluationMetrics(
        case_total=item["case_total"], case_passed=item["case_passed"],
        case_failed=item["case_failed"], case_errors=item["case_errors"],
        behavior_checks_total=item["behavior_checks_total"],
        behavior_checks_passed=item["behavior_checks_passed"],
        behavior_checks_failed=item["behavior_checks_failed"],
        case_pass_rate=item["case_pass_rate"],
        evaluation_error_rate=item["evaluation_error_rate"],
        behavior_check_pass_rate=item["behavior_check_pass_rate"],
        dimensions=dimensions,
    )


def _parse_dimension(payload: Any) -> DimensionMetrics:
    item = _fields(payload, {
        "dimension", "case_total", "case_passed", "case_failed", "case_errors",
        "behavior_checks_total", "behavior_checks_passed",
        "behavior_checks_failed", "case_pass_rate", "evaluation_error_rate",
        "behavior_check_pass_rate",
    })
    return DimensionMetrics(
        dimension=EvaluationDimension(item["dimension"]),
        case_total=item["case_total"], case_passed=item["case_passed"],
        case_failed=item["case_failed"], case_errors=item["case_errors"],
        behavior_checks_total=item["behavior_checks_total"],
        behavior_checks_passed=item["behavior_checks_passed"],
        behavior_checks_failed=item["behavior_checks_failed"],
        case_pass_rate=item["case_pass_rate"],
        evaluation_error_rate=item["evaluation_error_rate"],
        behavior_check_pass_rate=item["behavior_check_pass_rate"],
    )


def _validate_recalculated_metrics(report: EvaluationReport) -> None:
    results = tuple(
        EvaluationResult(
            case_id=scenario.case_id,
            status=scenario.status,
            checks=tuple(
                EvaluationCheck(name=check.name, passed=check.passed)
                for check in scenario.checks
            ),
            duration_ms=0.0,
        )
        for scenario in report.scenarios
    )
    expected = calculate_metrics(results)
    if any(not math.isfinite(value) for value in (
        report.metrics.case_pass_rate,
        report.metrics.evaluation_error_rate,
    )):
        raise EvaluationReportValidationError("Evaluation report metrics are invalid")
    if report.metrics.case_total != expected.case_total or _overall_tuple(report.metrics) != _overall_tuple(expected):
        raise EvaluationReportValidationError("Evaluation report metrics are inconsistent")
    expected_by_dimension = {item.dimension: item for item in expected.dimensions}
    reported_dimensions = {
        item.dimension for item in report.metrics.dimensions
    }
    if reported_dimensions != set(expected_by_dimension):
        raise EvaluationReportValidationError(
            "Evaluation report dimensions are inconsistent"
        )
    for dimension in report.metrics.dimensions:
        if dimension != expected_by_dimension[dimension.dimension]:
            raise EvaluationReportValidationError("Evaluation report metrics are inconsistent")


def _overall_tuple(metrics: EvaluationMetrics) -> tuple[Any, ...]:
    return (
        metrics.case_passed, metrics.case_failed, metrics.case_errors,
        metrics.behavior_checks_total, metrics.behavior_checks_passed,
        metrics.behavior_checks_failed, metrics.case_pass_rate,
        metrics.evaluation_error_rate, metrics.behavior_check_pass_rate,
    )
