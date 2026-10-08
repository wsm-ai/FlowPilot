import json
import math
from typing import Any

from app.evaluation.metrics import DimensionMetrics, EvaluationMetrics
from app.evaluation.report_models import EvaluationReport


def render_evaluation_report_json(report: EvaluationReport) -> str:
    payload = _report_payload(report)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation report cannot be serialized safely") from exc


def render_evaluation_report_markdown(report: EvaluationReport) -> str:
    metrics = report.metrics
    behavior_rate = _rate(metrics.behavior_check_pass_rate)
    lines = [
        "# FlowPilot Evaluation Report",
        "",
        "## Evaluation Summary",
        "",
        f"- Suite: `{_markdown(report.suite_name)}`",
        f"- Schema version: `{report.schema_version}`",
        f"- Total scenarios: {report.total_scenarios}",
        f"- Passed scenarios: {report.passed_scenarios}",
        f"- Failed scenarios: {report.failed_scenarios}",
        f"- Evaluation errors: {report.error_scenarios}",
        f"- Scenario pass rate: {_rate(metrics.case_pass_rate)}",
        "",
        "## Metrics Overview",
        "",
        "| Dimension | Cases | Passed | Failed | Errors | Case pass rate | Behavior check pass rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _dimension_row("overall", metrics),
    ]
    lines.extend(_dimension_row(item.dimension.value, item) for item in metrics.dimensions)
    lines.extend([
        "",
        f"Overall behavior check pass rate: {behavior_rate}",
        "",
        "## Scenario Results",
        "",
        "| Scenario ID | Status | Checks passed | Checks failed | Failure code |",
        "|---|---|---:|---:|---|",
    ])
    for scenario in report.scenarios:
        passed = sum(check.passed for check in scenario.checks)
        failed = len(scenario.checks) - passed
        failure = scenario.failure_code.value if scenario.failure_code else "—"
        lines.append(
            f"| {_markdown(scenario.case_id)} | {scenario.status.value} | "
            f"{passed} | {failed} | {failure} |"
        )
    if not report.scenarios:
        lines.append("| — | no scenarios | 0 | 0 | — |")
    lines.extend([
        "",
        "## Failure Analysis",
        "",
    ])
    failed_scenarios = [
        scenario for scenario in report.scenarios if scenario.failure_code is not None
    ]
    if failed_scenarios:
        lines.extend(
            f"- `{_markdown(item.case_id)}`: `{item.failure_code.value}`"
            for item in failed_scenarios
        )
    else:
        lines.append("No failed or errored scenarios were reported.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- This report uses deterministic scenarios and fake dependencies.",
        "- Results do not establish the accuracy of a live production model.",
        "- Capabilities without applicable metrics are not marked as verified.",
        "- No live DeepSeek, GitHub, MCP, or other remote service was called.",
        "",
    ])
    return "\n".join(lines)


def _report_payload(report: EvaluationReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "suite_name": report.suite_name,
        "total_scenarios": report.total_scenarios,
        "passed_scenarios": report.passed_scenarios,
        "failed_scenarios": report.failed_scenarios,
        "error_scenarios": report.error_scenarios,
        "metrics": _metrics_payload(report.metrics),
        "scenarios": [
            {
                "case_id": item.case_id,
                "status": item.status.value,
                "checks": [
                    {"name": check.name, "passed": check.passed}
                    for check in item.checks
                ],
                "failure_code": (
                    item.failure_code.value if item.failure_code else None
                ),
            }
            for item in report.scenarios
        ],
    }


def _metrics_payload(metrics: EvaluationMetrics) -> dict[str, Any]:
    return {
        "case_total": metrics.case_total,
        "case_passed": metrics.case_passed,
        "case_failed": metrics.case_failed,
        "case_errors": metrics.case_errors,
        "behavior_checks_total": metrics.behavior_checks_total,
        "behavior_checks_passed": metrics.behavior_checks_passed,
        "behavior_checks_failed": metrics.behavior_checks_failed,
        "case_pass_rate": _finite(metrics.case_pass_rate),
        "evaluation_error_rate": _finite(metrics.evaluation_error_rate),
        "behavior_check_pass_rate": _finite(metrics.behavior_check_pass_rate),
        "dimensions": [
            {
                "dimension": item.dimension.value,
                "case_total": item.case_total,
                "case_passed": item.case_passed,
                "case_failed": item.case_failed,
                "case_errors": item.case_errors,
                "behavior_checks_total": item.behavior_checks_total,
                "behavior_checks_passed": item.behavior_checks_passed,
                "behavior_checks_failed": item.behavior_checks_failed,
                "case_pass_rate": _finite(item.case_pass_rate),
                "evaluation_error_rate": _finite(item.evaluation_error_rate),
                "behavior_check_pass_rate": _finite(item.behavior_check_pass_rate),
            }
            for item in metrics.dimensions
        ],
    }


def _finite(value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError("Evaluation report contains a non-finite metric")
    return value


def _rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _dimension_row(name: str, metrics: EvaluationMetrics | DimensionMetrics) -> str:
    return (
        f"| {name} | {metrics.case_total} | {metrics.case_passed} | "
        f"{metrics.case_failed} | {metrics.case_errors} | "
        f"{_rate(metrics.case_pass_rate)} | "
        f"{_rate(metrics.behavior_check_pass_rate)} |"
    )


def _markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")
