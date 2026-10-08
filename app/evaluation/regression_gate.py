from app.evaluation.gate_models import (
    GateCheck,
    GateReason,
    GateStatus,
    RegressionGateConfig,
    RegressionGateResult,
)
from app.evaluation.report_models import EvaluationReport


def evaluate_regression_gate(
    report: EvaluationReport,
    *,
    config: RegressionGateConfig | None = None,
    baseline: EvaluationReport | None = None,
) -> RegressionGateResult:
    policy = config or RegressionGateConfig()
    checks: list[GateCheck] = []

    _add(
        checks,
        "non_empty_evaluation",
        report.total_scenarios > 0,
        GateReason.EMPTY_EVALUATION,
    )
    _add(
        checks,
        "failed_scenario_limit",
        report.failed_scenarios <= policy.max_failed_scenarios,
        GateReason.FAILED_SCENARIOS,
    )
    _add(
        checks,
        "evaluation_error_limit",
        report.error_scenarios <= policy.max_error_scenarios,
        GateReason.EVALUATION_ERRORS,
    )
    _add(
        checks,
        "case_pass_rate_threshold",
        report.metrics.case_pass_rate >= policy.min_case_pass_rate,
        GateReason.BELOW_THRESHOLD,
    )
    behavior_rate = report.metrics.behavior_check_pass_rate
    _add(
        checks,
        "behavior_check_metric_available",
        behavior_rate is not None,
        GateReason.MISSING_REQUIRED_METRIC,
    )
    if behavior_rate is not None:
        _add(
            checks,
            "behavior_check_pass_rate_threshold",
            behavior_rate >= policy.min_behavior_check_pass_rate,
            GateReason.BELOW_THRESHOLD,
        )

    by_dimension = {
        item.dimension: item for item in report.metrics.dimensions
    }
    for threshold in policy.dimension_thresholds:
        metric = by_dimension.get(threshold.dimension)
        _add(
            checks,
            f"dimension_{threshold.dimension.value}_available",
            metric is not None,
            GateReason.MISSING_REQUIRED_DIMENSION,
        )
        if metric is None:
            continue
        _add(
            checks,
            f"dimension_{threshold.dimension.value}_applicable",
            metric.case_total > 0,
            GateReason.MISSING_REQUIRED_METRIC,
        )
        if metric.case_total > 0:
            _add(
                checks,
                f"dimension_{threshold.dimension.value}_threshold",
                metric.case_pass_rate >= threshold.min_case_pass_rate,
                GateReason.BELOW_THRESHOLD,
            )

    baseline_compared = False
    if baseline is not None:
        incompatible = _incompatible(report, baseline)
        if incompatible:
            checks.append(GateCheck(
                name="baseline_compatibility",
                passed=False,
                reason=GateReason.INCOMPATIBLE_BASELINE,
            ))
            return _result(
                GateStatus.INVALID, report, checks, baseline_compared=False
            )
        checks.append(GateCheck(name="baseline_compatibility", passed=True))
        baseline_compared = True
        _regression_checks(checks, report, baseline, policy.max_regression)

    status = (
        GateStatus.PASS
        if all(check.passed for check in checks)
        else GateStatus.FAIL
    )
    return _result(status, report, checks, baseline_compared)


def _incompatible(current: EvaluationReport, baseline: EvaluationReport) -> bool:
    if (
        current.schema_version != baseline.schema_version
        or current.suite_name != baseline.suite_name
    ):
        return True
    current_structure = {
        item.case_id: tuple(check.name for check in item.checks)
        for item in current.scenarios
    }
    baseline_structure = {
        item.case_id: tuple(check.name for check in item.checks)
        for item in baseline.scenarios
    }
    return current_structure != baseline_structure


def _regression_checks(
    checks: list[GateCheck],
    current: EvaluationReport,
    baseline: EvaluationReport,
    tolerance: float,
) -> None:
    _add(
        checks,
        "overall_case_pass_rate_regression",
        current.metrics.case_pass_rate + tolerance
        >= baseline.metrics.case_pass_rate,
        GateReason.REGRESSION_DETECTED,
    )
    current_behavior = current.metrics.behavior_check_pass_rate
    baseline_behavior = baseline.metrics.behavior_check_pass_rate
    if (current_behavior is None) != (baseline_behavior is None):
        _add(
            checks,
            "overall_behavior_metric_comparable",
            False,
            GateReason.INCOMPATIBLE_BASELINE,
        )
    elif current_behavior is not None and baseline_behavior is not None:
        _add(
            checks,
            "overall_behavior_pass_rate_regression",
            current_behavior + tolerance >= baseline_behavior,
            GateReason.REGRESSION_DETECTED,
        )
    current_dimensions = {
        item.dimension: item for item in current.metrics.dimensions
    }
    baseline_dimensions = {
        item.dimension: item for item in baseline.metrics.dimensions
    }
    for dimension in sorted(
        set(current_dimensions) & set(baseline_dimensions),
        key=lambda item: item.value,
    ):
        current_metric = current_dimensions[dimension]
        baseline_metric = baseline_dimensions[dimension]
        if current_metric.case_total == 0 and baseline_metric.case_total == 0:
            continue
        _add(
            checks,
            f"dimension_{dimension.value}_case_rate_regression",
            current_metric.case_pass_rate + tolerance
            >= baseline_metric.case_pass_rate,
            GateReason.REGRESSION_DETECTED,
        )
        current_rate = current_metric.behavior_check_pass_rate
        baseline_rate = baseline_metric.behavior_check_pass_rate
        if (current_rate is None) != (baseline_rate is None):
            _add(
                checks,
                f"dimension_{dimension.value}_behavior_comparable",
                False,
                GateReason.INCOMPATIBLE_BASELINE,
            )
        elif current_rate is not None and baseline_rate is not None:
            _add(
                checks,
                f"dimension_{dimension.value}_behavior_regression",
                current_rate + tolerance >= baseline_rate,
                GateReason.REGRESSION_DETECTED,
            )


def _add(
    checks: list[GateCheck],
    name: str,
    passed: bool,
    reason: GateReason,
) -> None:
    checks.append(GateCheck(
        name=name,
        passed=passed,
        reason=None if passed else reason,
    ))


def _result(
    status: GateStatus,
    report: EvaluationReport,
    checks: list[GateCheck],
    baseline_compared: bool,
) -> RegressionGateResult:
    reasons = tuple(dict.fromkeys(
        check.reason for check in checks if check.reason is not None
    ))
    return RegressionGateResult(
        gate_status=status,
        total_scenarios=report.total_scenarios,
        passed_scenarios=report.passed_scenarios,
        failed_scenarios=report.failed_scenarios,
        error_scenarios=report.error_scenarios,
        baseline_compared=baseline_compared,
        checks=tuple(checks),
        reasons=reasons,
    )
