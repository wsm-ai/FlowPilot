from dataclasses import replace
import json
from pathlib import Path

import pytest

from app.evaluation.gate_cli import main as gate_main
from app.evaluation.gate_models import (
    DimensionThreshold,
    GateReason,
    GateStatus,
    RegressionGateConfig,
    RegressionGateConfigurationError,
)
from app.evaluation.gate_validator import (
    EvaluationReportValidationError,
    load_evaluation_report,
)
from app.evaluation.metrics import EvaluationDimension
from app.evaluation.models import EvaluationCheck, EvaluationResult, EvaluationStatus
from app.evaluation.regression_gate import evaluate_regression_gate
from app.evaluation.report_builder import build_evaluation_report
from app.evaluation.report_renderers import render_evaluation_report_json


def _result(
    case_id: str,
    status: EvaluationStatus = EvaluationStatus.PASSED,
    checks: tuple[bool, ...] = (True,),
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        status=status,
        checks=tuple(
            EvaluationCheck(name=f"check_{index}", passed=passed)
            for index, passed in enumerate(checks)
        ),
        duration_ms=0,
    )


def _report(*results: EvaluationResult):
    return build_evaluation_report(results, suite_name="gate-suite")


def _write(path: Path, report) -> Path:
    path.write_text(render_evaluation_report_json(report), encoding="utf-8")
    return path


def test_real_11f_report_passes_default_gate(tmp_path: Path) -> None:
    report = _report(
        _result("planner.one"),
        _result("hitl.one"),
        _result("rag.one"),
        _result("mcp.one"),
        _result("reliability.one"),
    )
    loaded = load_evaluation_report(_write(tmp_path / "report.json", report))
    result = evaluate_regression_gate(loaded)
    assert result.gate_status is GateStatus.PASS
    assert not result.baseline_compared and result.reasons == ()


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (EvaluationStatus.FAILED, GateReason.FAILED_SCENARIOS),
        (EvaluationStatus.ERROR, GateReason.EVALUATION_ERRORS),
    ],
)
def test_failed_and_error_scenarios_fail(status, reason) -> None:
    result = evaluate_regression_gate(
        _report(_result("planner.one", status, (False,)))
    )
    assert result.gate_status is GateStatus.FAIL
    assert reason in result.reasons


def test_empty_and_missing_behavior_metric_cannot_pass() -> None:
    result = evaluate_regression_gate(_report())
    assert result.gate_status is GateStatus.FAIL
    assert GateReason.EMPTY_EVALUATION in result.reasons
    assert GateReason.MISSING_REQUIRED_METRIC in result.reasons


def test_configurable_case_and_behavior_thresholds() -> None:
    report = _report(
        _result("planner.pass"),
        _result("planner.fail", EvaluationStatus.FAILED, (True, False)),
    )
    strict = evaluate_regression_gate(report)
    relaxed = evaluate_regression_gate(
        report,
        config=RegressionGateConfig(
            max_failed_scenarios=1,
            min_case_pass_rate=0.5,
            min_behavior_check_pass_rate=2 / 3,
        ),
    )
    assert strict.gate_status is GateStatus.FAIL
    assert relaxed.gate_status is GateStatus.PASS


def test_required_dimension_missing_or_not_applicable_fails() -> None:
    report = _report(_result("planner.one"))
    threshold = RegressionGateConfig(dimension_thresholds=(
        DimensionThreshold(EvaluationDimension.RAG, 1.0),
    ))
    not_applicable = evaluate_regression_gate(report, config=threshold)
    assert GateReason.MISSING_REQUIRED_METRIC in not_applicable.reasons

    without_rag = replace(
        report,
        metrics=replace(
            report.metrics,
            dimensions=tuple(
                item for item in report.metrics.dimensions
                if item.dimension is not EvaluationDimension.RAG
            ),
        ),
    )
    missing = evaluate_regression_gate(without_rag, config=threshold)
    assert GateReason.MISSING_REQUIRED_DIMENSION in missing.reasons


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_failed_scenarios": -1},
        {"max_error_scenarios": True},
        {"min_case_pass_rate": -0.1},
        {"min_behavior_check_pass_rate": 1.1},
        {"max_regression": float("nan")},
        {"max_regression": float("inf")},
    ],
)
def test_invalid_gate_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(RegressionGateConfigurationError):
        RegressionGateConfig(**kwargs)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":"1.0","schema_version":"1.0"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"schema_version":"99.0"}',
    ],
)
def test_unsafe_or_invalid_json_is_rejected(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(EvaluationReportValidationError):
        load_evaluation_report(path)


def test_duplicate_scenario_and_inconsistent_counts_are_rejected(
    tmp_path: Path,
) -> None:
    payload = json.loads(render_evaluation_report_json(_report(_result("planner.one"))))
    payload["scenarios"].append(payload["scenarios"][0])
    payload["total_scenarios"] = 2
    payload["passed_scenarios"] = 2
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationReportValidationError):
        load_evaluation_report(duplicate)

    payload = json.loads(render_evaluation_report_json(_report(_result("planner.one"))))
    payload["passed_scenarios"] = 0
    inconsistent = tmp_path / "inconsistent.json"
    inconsistent.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationReportValidationError):
        load_evaluation_report(inconsistent)


def test_inconsistent_metrics_are_rejected(tmp_path: Path) -> None:
    payload = json.loads(render_evaluation_report_json(_report(_result("planner.one"))))
    payload["metrics"]["behavior_checks_passed"] = 0
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationReportValidationError, match="inconsistent"):
        load_evaluation_report(path)


@pytest.mark.parametrize("removed_dimension", ["planner", "hitl", "all"])
def test_missing_metric_dimensions_are_rejected(
    tmp_path: Path,
    removed_dimension: str,
) -> None:
    payload = json.loads(
        render_evaluation_report_json(_report(_result("planner.one")))
    )
    if removed_dimension == "all":
        payload["metrics"]["dimensions"] = []
    else:
        payload["metrics"]["dimensions"] = [
            item for item in payload["metrics"]["dimensions"]
            if item["dimension"] != removed_dimension
        ]
    path = tmp_path / f"missing-{removed_dimension}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        EvaluationReportValidationError,
        match="dimensions are inconsistent",
    ):
        load_evaluation_report(path)


def test_complete_dimensions_load_and_modified_dimension_is_rejected(
    tmp_path: Path,
) -> None:
    report = _report(_result("planner.one"))
    complete = _write(tmp_path / "complete.json", report)
    assert load_evaluation_report(complete) == report

    payload = json.loads(render_evaluation_report_json(report))
    planner = next(
        item for item in payload["metrics"]["dimensions"]
        if item["dimension"] == "planner"
    )
    planner["case_passed"] = 0
    modified = tmp_path / "modified.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationReportValidationError, match="inconsistent"):
        load_evaluation_report(modified)


def test_cli_marks_missing_dimension_report_invalid(
    tmp_path: Path,
    capsys,
) -> None:
    payload = json.loads(
        render_evaluation_report_json(_report(_result("planner.one")))
    )
    payload["metrics"]["dimensions"] = payload["metrics"]["dimensions"][1:]
    path = tmp_path / "missing-dimension.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert gate_main(["--report", str(path)]) == 2
    output = capsys.readouterr().out
    assert output == "INVALID\nreason: invalid_report\n"


def test_identical_baseline_passes_and_marks_comparison() -> None:
    report = _report(_result("planner.one"), _result("rag.one"))
    result = evaluate_regression_gate(report, baseline=report)
    assert result.gate_status is GateStatus.PASS
    assert result.baseline_compared


def test_quality_regression_is_detected_and_improvement_passes() -> None:
    baseline = _report(
        _result("planner.one"), _result("rag.one")
    )
    worse = _report(
        _result("planner.one"),
        _result("rag.one", EvaluationStatus.FAILED, (False,)),
    )
    relaxed = RegressionGateConfig(
        max_failed_scenarios=1,
        min_case_pass_rate=0,
        min_behavior_check_pass_rate=0,
    )
    result = evaluate_regression_gate(worse, baseline=baseline, config=relaxed)
    assert result.gate_status is GateStatus.FAIL
    assert GateReason.REGRESSION_DETECTED in result.reasons

    improved = evaluate_regression_gate(baseline, baseline=worse)
    assert improved.gate_status is GateStatus.PASS


@pytest.mark.parametrize(
    "baseline",
    [
        _report(_result("planner.other")),
        build_evaluation_report([_result("planner.one")], suite_name="other-suite"),
    ],
)
def test_changed_coverage_or_suite_is_incompatible(baseline) -> None:
    current = _report(_result("planner.one"))
    result = evaluate_regression_gate(current, baseline=baseline)
    assert result.gate_status is GateStatus.INVALID
    assert result.baseline_compared is False
    assert GateReason.INCOMPATIBLE_BASELINE in result.reasons


def test_dimension_regression_fails_when_overall_rate_is_unchanged() -> None:
    baseline = _report(
        _result("planner.one"),
        _result("rag.one", EvaluationStatus.FAILED, (False,)),
    )
    current = _report(
        _result("planner.one", EvaluationStatus.FAILED, (False,)),
        _result("rag.one"),
    )
    assert current.metrics.case_pass_rate == baseline.metrics.case_pass_rate
    assert (
        current.metrics.behavior_check_pass_rate
        == baseline.metrics.behavior_check_pass_rate
    )
    config = RegressionGateConfig(
        max_failed_scenarios=1,
        min_case_pass_rate=0,
        min_behavior_check_pass_rate=0,
    )
    result = evaluate_regression_gate(
        current,
        baseline=baseline,
        config=config,
    )
    assert result.gate_status is GateStatus.FAIL
    assert result.baseline_compared is True
    failed_checks = {check.name for check in result.checks if not check.passed}
    assert "dimension_planner_case_rate_regression" in failed_checks
    assert "overall_case_pass_rate_regression" not in failed_checks


def test_regression_tolerance_is_honored() -> None:
    baseline = _report(
        _result("planner.one"), _result("planner.two")
    )
    current = _report(
        _result("planner.one"),
        _result("planner.two", EvaluationStatus.FAILED, (False,)),
    )
    config = RegressionGateConfig(
        max_failed_scenarios=1,
        min_case_pass_rate=0,
        min_behavior_check_pass_rate=0,
        max_regression=0.5,
    )
    assert evaluate_regression_gate(
        current, baseline=baseline, config=config
    ).gate_status is GateStatus.PASS


def test_cli_exit_codes_and_safe_output(tmp_path: Path, capsys) -> None:
    passing = _write(tmp_path / "passing.json", _report(_result("planner.one")))
    failing = _write(
        tmp_path / "failing.json",
        _report(_result("planner.one", EvaluationStatus.FAILED, (False,))),
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"TOKEN":"super-secret"}', encoding="utf-8")

    assert gate_main(["--report", str(passing)]) == 0
    assert gate_main(["--report", str(failing)]) == 1
    assert gate_main(["--report", str(invalid)]) == 2
    output = capsys.readouterr().out
    assert "PASS" in output and "FAIL" in output and "INVALID" in output
    assert "super-secret" not in output and "TOKEN" not in output


def test_gate_is_deterministic_and_does_not_modify_report(tmp_path: Path) -> None:
    path = _write(tmp_path / "report.json", _report(_result("mcp.one")))
    before = path.read_bytes()
    first = evaluate_regression_gate(load_evaluation_report(path))
    second = evaluate_regression_gate(load_evaluation_report(path))
    assert first == second
    assert path.read_bytes() == before
