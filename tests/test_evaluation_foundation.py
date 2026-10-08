import asyncio
from dataclasses import FrozenInstanceError

import pytest

from app.evaluation import (
    EvaluationCase,
    EvaluationCheck,
    EvaluationConfigurationError,
    EvaluationResult,
    EvaluationRunner,
    EvaluationStatus,
    summarize,
)


class Scenario:
    def __init__(self, checks=(), error=None):
        self.checks = checks
        self.error = error
        self.call_count = 0

    async def run(self):
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.checks


def case(case_id, scenario, name="Evaluation case"):
    return EvaluationCase(
        case_id=case_id,
        name=name,
        description="Deterministic offline scenario",
        scenario=scenario,
    )


def test_all_checks_pass():
    checks = [
        EvaluationCheck("expected_tool_selected", True),
        EvaluationCheck("no_forbidden_action", True),
    ]
    result = asyncio.run(EvaluationRunner().run([case("case-1", Scenario(checks))]))[0]
    assert result.status is EvaluationStatus.PASSED
    assert result.checks == tuple(checks)


def test_one_failed_check_makes_case_failed_and_preserves_check_order():
    checks = [
        EvaluationCheck("first", True),
        EvaluationCheck("second", False, "Expected action was not selected"),
        EvaluationCheck("third", True),
    ]
    result = asyncio.run(EvaluationRunner().run([case("case-1", Scenario(checks))]))[0]
    assert result.status is EvaluationStatus.FAILED
    assert [check.name for check in result.checks] == ["first", "second", "third"]


def test_multiple_case_results_preserve_input_order():
    cases = [
        case("case-b", Scenario([EvaluationCheck("check", True)])),
        case("case-a", Scenario([EvaluationCheck("check", False)])),
    ]
    results = asyncio.run(EvaluationRunner().run(cases))
    assert [result.case_id for result in results] == ["case-b", "case-a"]


def test_duplicate_case_id_fails_before_any_scenario_runs():
    first = Scenario()
    second = Scenario()
    with pytest.raises(EvaluationConfigurationError, match="Duplicate"):
        asyncio.run(
            EvaluationRunner().run(
                [case("duplicate", first), case("duplicate", second)]
            )
        )
    assert first.call_count == second.call_count == 0


@pytest.mark.parametrize("case_id", ["", "   "])
def test_empty_case_id_is_rejected(case_id):
    with pytest.raises(EvaluationConfigurationError, match="case_id"):
        case(case_id, Scenario())


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_check_name_is_rejected(name):
    with pytest.raises(EvaluationConfigurationError, match="check name"):
        EvaluationCheck(name, True)


def test_unexpected_exception_becomes_safe_error_without_retry():
    scenario = Scenario(error=RuntimeError("Bearer SUPER_SECRET"))
    result = asyncio.run(EvaluationRunner().run([case("case-error", scenario)]))[0]
    assert result.status is EvaluationStatus.ERROR
    assert result.checks == (
        EvaluationCheck(
            "scenario_execution", False, "Evaluation scenario failed"
        ),
    )
    assert "SUPER_SECRET" not in repr(result)
    assert scenario.call_count == 1


def test_cancelled_error_propagates():
    scenario = Scenario(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(EvaluationRunner().run([case("cancelled", scenario)]))
    assert scenario.call_count == 1


def test_failed_and_error_cases_do_not_stop_remaining_cases():
    final = Scenario([EvaluationCheck("final", True)])
    results = asyncio.run(
        EvaluationRunner().run(
            [
                case("failed", Scenario([EvaluationCheck("check", False)])),
                case("error", Scenario(error=RuntimeError("private"))),
                case("passed", final),
            ]
        )
    )
    assert [result.status for result in results] == [
        EvaluationStatus.FAILED,
        EvaluationStatus.ERROR,
        EvaluationStatus.PASSED,
    ]
    assert final.call_count == 1


def test_zero_checks_is_safe_error_and_does_not_stop_following_case():
    following = Scenario([EvaluationCheck("following", True)])
    results = asyncio.run(
        EvaluationRunner().run(
            [
                case("zero-checks", Scenario([])),
                case("following", following),
            ]
        )
    )
    assert results[0].status is EvaluationStatus.ERROR
    assert results[0].status is not EvaluationStatus.PASSED
    assert results[0].checks == (
        EvaluationCheck(
            "scenario_execution",
            False,
            "Evaluation scenario produced no checks",
        ),
    )
    assert results[1].status is EvaluationStatus.PASSED
    assert following.call_count == 1


def test_summary_counts_and_pass_rate_are_deterministic():
    results = asyncio.run(
        EvaluationRunner().run(
            [
                case("passed", Scenario([EvaluationCheck("check", True)])),
                case("failed", Scenario([EvaluationCheck("check", False)])),
                case("error", Scenario(error=RuntimeError("private"))),
            ]
        )
    )
    summary = summarize(results)
    assert (summary.total, summary.passed, summary.failed, summary.errors) == (
        3, 1, 1, 1
    )
    assert summary.pass_rate == pytest.approx(1 / 3)


def test_empty_summary_has_zero_pass_rate():
    summary = summarize([])
    assert (summary.total, summary.passed, summary.failed, summary.errors) == (
        0, 0, 0, 0
    )
    assert summary.pass_rate == 0.0


def test_duration_is_non_negative_measurement():
    result = asyncio.run(EvaluationRunner().run([case("timed", Scenario())]))[0]
    assert result.duration_ms >= 0


@pytest.mark.parametrize(
    "duration_ms", [float("nan"), float("inf"), float("-inf"), -0.1]
)
def test_duration_must_be_finite_and_non_negative(duration_ms):
    with pytest.raises(EvaluationConfigurationError, match="finite"):
        EvaluationResult(
            case_id="invalid-duration",
            status=EvaluationStatus.ERROR,
            checks=(),
            duration_ms=duration_ms,
        )


def test_models_are_immutable_and_case_repr_omits_scenario():
    secret_scenario = Scenario(error=RuntimeError("SUPER_SECRET"))
    evaluation_case = case("immutable", secret_scenario)
    check = EvaluationCheck("immutable", True)
    with pytest.raises(FrozenInstanceError):
        check.passed = False
    assert "SUPER_SECRET" not in repr(evaluation_case)
