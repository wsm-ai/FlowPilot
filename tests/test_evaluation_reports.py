import asyncio
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest
import app.evaluation.report_writer as report_writer_module

from app.evaluation import (
    EvaluationCheck,
    EvaluationReportConfigurationError,
    EvaluationReportCheck,
    EvaluationScenarioReport,
    EvaluationReportWriteError,
    EvaluationResult,
    EvaluationRunner,
    EvaluationStatus,
    ReportFailureCode,
    build_default_evaluation_cases,
    build_evaluation_report,
    render_evaluation_report_json,
    render_evaluation_report_markdown,
    write_evaluation_reports,
)
from app.evaluation.cli import main


def _result(
    case_id: str,
    status: EvaluationStatus,
    checks: tuple[bool, ...] = (True,),
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        status=status,
        checks=tuple(
            EvaluationCheck(
                name=f"check_{index}",
                passed=passed,
                message="TOKEN=must-not-be-reported" if not passed else None,
            )
            for index, passed in enumerate(checks)
        ),
        duration_ms=123.456,
    )


def test_report_builds_from_real_results_and_is_immutable() -> None:
    results = asyncio.run(
        EvaluationRunner().run(build_default_evaluation_cases())
    )
    report = build_evaluation_report(results)
    assert report.total_scenarios == len(results)
    assert report.passed_scenarios == len(results)
    assert report.failed_scenarios == report.error_scenarios == 0
    with pytest.raises(FrozenInstanceError):
        report.total_scenarios = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.scenarios[0].case_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ((EvaluationStatus.PASSED,), (1, 0, 0)),
        ((EvaluationStatus.PASSED, EvaluationStatus.FAILED), (1, 1, 0)),
        ((EvaluationStatus.FAILED, EvaluationStatus.FAILED), (0, 2, 0)),
        ((EvaluationStatus.ERROR,), (0, 0, 1)),
    ],
)
def test_status_aggregates_are_exact(statuses, expected) -> None:
    report = build_evaluation_report(
        [
            _result(f"planner.case_{index}", status, (status is EvaluationStatus.PASSED,))
            for index, status in enumerate(statuses)
        ]
    )
    assert (
        report.passed_scenarios,
        report.failed_scenarios,
        report.error_scenarios,
    ) == expected


def test_empty_report_is_explicit_and_metrics_are_not_fabricated() -> None:
    report = build_evaluation_report([])
    assert report.total_scenarios == 0
    assert report.metrics.case_pass_rate == 0.0
    assert report.metrics.behavior_check_pass_rate is None
    assert "N/A" in render_evaluation_report_markdown(report)


def test_rendering_is_stably_sorted_and_byte_deterministic() -> None:
    results = [
        _result("rag.zeta", EvaluationStatus.FAILED, (True, False)),
        _result("planner.alpha", EvaluationStatus.PASSED),
    ]
    first = build_evaluation_report(results)
    second = build_evaluation_report(list(reversed(results)))
    assert [item.case_id for item in first.scenarios] == [
        "planner.alpha",
        "rag.zeta",
    ]
    assert render_evaluation_report_json(first) == render_evaluation_report_json(second)
    assert render_evaluation_report_markdown(first) == render_evaluation_report_markdown(second)


def test_json_is_standard_safe_and_contains_existing_metrics() -> None:
    rendered = render_evaluation_report_json(
        build_evaluation_report([_result("hitl.safe", EvaluationStatus.PASSED)])
    )
    payload = json.loads(rendered)
    assert payload["schema_version"] == "1.0"
    assert payload["metrics"]["dimensions"][1]["dimension"] == "hitl"
    assert payload["scenarios"][0]["status"] == "passed"
    assert "NaN" not in rendered and "Infinity" not in rendered


def test_safe_failure_codes_replace_messages_and_sensitive_text() -> None:
    rendered = render_evaluation_report_json(
        build_evaluation_report([
            _result("mcp.failure", EvaluationStatus.FAILED, (False,)),
            _result("rag.error", EvaluationStatus.ERROR, (False,)),
        ])
    )
    assert "check_failed" in rendered
    assert "scenario_error" in rendered
    assert "TOKEN=" not in rendered
    assert "123.456" not in rendered


def test_invalid_status_and_unsafe_identifier_are_rejected() -> None:
    invalid = _result("planner.invalid", EvaluationStatus.PASSED)
    object.__setattr__(invalid, "status", "unknown")
    with pytest.raises(Exception, match="invalid status"):
        build_evaluation_report([invalid])
    unsafe = _result("planner.safe", EvaluationStatus.PASSED)
    object.__setattr__(unsafe, "case_id", "planner.secret|TOKEN")
    with pytest.raises(EvaluationReportConfigurationError, match="safe report"):
        build_evaluation_report([unsafe])


def test_report_does_not_mutate_source_results() -> None:
    result = _result("reliability.original", EvaluationStatus.FAILED, (False,))
    original = (result.case_id, result.status, result.checks, result.duration_ms)
    build_evaluation_report([result])
    assert (result.case_id, result.status, result.checks, result.duration_ms) == original


def test_writer_creates_utf8_artifacts_and_failed_staging_preserves_existing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "reports"
    json_path, markdown_path = write_evaluation_reports(
        output, json_report='{"suite":"测试"}\n', markdown_report="# Report\n"
    )
    assert json_path.read_text(encoding="utf-8") == '{"suite":"测试"}\n'
    assert markdown_path.read_text(encoding="utf-8") == "# Report\n"
    with pytest.raises(EvaluationReportWriteError):
        write_evaluation_reports(
            output,
            json_report="replacement",
            markdown_report=None,  # type: ignore[arg-type]
        )
    assert json_path.read_text(encoding="utf-8") == '{"suite":"测试"}\n'
    assert not list(output.glob("*.tmp"))


def test_cli_runs_offline_and_writes_fixed_filenames(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    assert main(["--output-dir", str(output)]) == 0
    json_path = output / "evaluation-report.json"
    markdown_path = output / "evaluation-report.md"
    assert json_path.is_file() and markdown_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total_scenarios"] == len(build_default_evaluation_cases())
    assert "# FlowPilot Evaluation Report" in markdown_path.read_text(
        encoding="utf-8"
    )


def test_cli_invalid_suite_fails_safely(tmp_path: Path, capsys) -> None:
    assert main([
        "--output-dir", str(tmp_path), "--suite-name", "bad|suite"
    ]) == 1
    captured = capsys.readouterr()
    assert captured.err == "Evaluation report generation failed\n"


@pytest.mark.parametrize("targets_exist", [True, False])
def test_second_replace_failure_restores_both_report_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    targets_exist: bool,
) -> None:
    json_path = tmp_path / "evaluation-report.json"
    markdown_path = tmp_path / "evaluation-report.md"
    if targets_exist:
        json_path.write_text("old-json", encoding="utf-8")
        markdown_path.write_text("old-markdown", encoding="utf-8")

    real_replace = report_writer_module.os.replace
    calls = 0

    def fail_second_replace(source, target) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replacement failure")
        real_replace(source, target)

    monkeypatch.setattr(report_writer_module.os, "replace", fail_second_replace)
    with pytest.raises(
        EvaluationReportWriteError,
        match="Unable to write evaluation reports",
    ):
        write_evaluation_reports(
            tmp_path,
            json_report="new-json",
            markdown_report="new-markdown",
        )

    if targets_exist:
        assert json_path.read_text(encoding="utf-8") == "old-json"
        assert markdown_path.read_text(encoding="utf-8") == "old-markdown"
        assert set(tmp_path.iterdir()) == {json_path, markdown_path}
    else:
        assert not json_path.exists()
        assert not markdown_path.exists()
        assert list(tmp_path.iterdir()) == []


def test_report_rejects_duplicate_scenarios_and_incorrect_status_counts() -> None:
    report = build_evaluation_report([
        _result("planner.one", EvaluationStatus.PASSED),
        _result("rag.two", EvaluationStatus.FAILED, (False,)),
    ])
    with pytest.raises(EvaluationReportConfigurationError, match="duplicate"):
        replace(report, scenarios=(report.scenarios[0], report.scenarios[0]))
    for field_name in (
        "passed_scenarios",
        "failed_scenarios",
        "error_scenarios",
    ):
        with pytest.raises(
            EvaluationReportConfigurationError,
            match="status counts",
        ):
            replace(report, **{field_name: getattr(report, field_name) + 1})


@pytest.mark.parametrize(
    ("status", "failure_code"),
    [
        (EvaluationStatus.PASSED, ReportFailureCode.CHECK_FAILED),
        (EvaluationStatus.FAILED, None),
        (EvaluationStatus.FAILED, ReportFailureCode.SCENARIO_ERROR),
        (EvaluationStatus.ERROR, None),
        (EvaluationStatus.ERROR, ReportFailureCode.CHECK_FAILED),
    ],
)
def test_scenario_report_rejects_inconsistent_failure_code(
    status: EvaluationStatus,
    failure_code: ReportFailureCode | None,
) -> None:
    with pytest.raises(EvaluationReportConfigurationError, match="inconsistent"):
        EvaluationScenarioReport(
            case_id="planner.safe",
            status=status,
            checks=(EvaluationReportCheck("safe_check", status is EvaluationStatus.PASSED),),
            failure_code=failure_code,
        )


def test_scenario_report_rejects_invalid_check_element() -> None:
    with pytest.raises(EvaluationReportConfigurationError, match="invalid check"):
        EvaluationScenarioReport(
            case_id="planner.safe",
            status=EvaluationStatus.PASSED,
            checks=(object(),),  # type: ignore[arg-type]
            failure_code=None,
        )
