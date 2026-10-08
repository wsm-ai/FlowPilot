from collections.abc import Sequence

from app.evaluation.metrics import calculate_metrics
from app.evaluation.models import EvaluationResult, EvaluationStatus
from app.evaluation.report_models import (
    REPORT_SCHEMA_VERSION,
    EvaluationReport,
    EvaluationReportCheck,
    EvaluationScenarioReport,
    ReportFailureCode,
)


def build_evaluation_report(
    results: Sequence[EvaluationResult],
    *,
    suite_name: str = "flowpilot-default",
) -> EvaluationReport:
    """Build an immutable report without re-running evaluation scenarios."""
    ordered_results = tuple(sorted(results, key=lambda item: item.case_id))
    metrics = calculate_metrics(ordered_results)
    scenarios = tuple(_scenario_report(result) for result in ordered_results)
    return EvaluationReport(
        schema_version=REPORT_SCHEMA_VERSION,
        suite_name=suite_name,
        total_scenarios=metrics.case_total,
        passed_scenarios=metrics.case_passed,
        failed_scenarios=metrics.case_failed,
        error_scenarios=metrics.case_errors,
        metrics=metrics,
        scenarios=scenarios,
    )


def _scenario_report(result: EvaluationResult) -> EvaluationScenarioReport:
    if result.status is EvaluationStatus.PASSED:
        failure_code = None
    elif result.status is EvaluationStatus.FAILED:
        failure_code = ReportFailureCode.CHECK_FAILED
    else:
        failure_code = ReportFailureCode.SCENARIO_ERROR
    return EvaluationScenarioReport(
        case_id=result.case_id,
        status=result.status,
        checks=tuple(
            EvaluationReportCheck(name=check.name, passed=check.passed)
            for check in result.checks
        ),
        failure_code=failure_code,
    )
