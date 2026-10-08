import asyncio
from dataclasses import FrozenInstanceError

import pytest

from app.evaluation import (
    EvaluationCheck,
    EvaluationDimension,
    EvaluationMetricConfigurationError,
    EvaluationResult,
    EvaluationRunner,
    EvaluationStatus,
    build_default_evaluation_cases,
    calculate_metrics,
)


def _result(
    case_id: str,
    status: EvaluationStatus,
    passed: tuple[bool, ...],
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        status=status,
        checks=tuple(
            EvaluationCheck(name=f"check_{index}", passed=value)
            for index, value in enumerate(passed)
        ),
        duration_ms=1.0,
    )


def test_all_passed_results() -> None:
    metrics = calculate_metrics([
        _result("planner.one", EvaluationStatus.PASSED, (True, True)),
        _result("hitl.one", EvaluationStatus.PASSED, (True,)),
    ])
    assert metrics.case_total == 2
    assert metrics.case_passed == 2
    assert metrics.case_failed == 0
    assert metrics.case_errors == 0
    assert metrics.case_pass_rate == 1.0
    assert metrics.evaluation_error_rate == 0.0
    assert metrics.behavior_check_pass_rate == 1.0


def test_mixed_status_counts_and_behavior_denominators() -> None:
    metrics = calculate_metrics([
        _result("planner.pass", EvaluationStatus.PASSED, (True, True)),
        _result("hitl.fail", EvaluationStatus.FAILED, (True, True, True, False)),
        _result("rag.error", EvaluationStatus.ERROR, (False,)),
    ])
    assert (metrics.case_total, metrics.case_passed) == (3, 1)
    assert (metrics.case_failed, metrics.case_errors) == (1, 1)
    assert metrics.case_pass_rate == pytest.approx(1 / 3)
    assert metrics.evaluation_error_rate == pytest.approx(1 / 3)
    assert metrics.behavior_checks_total == 6
    assert metrics.behavior_checks_passed == 5
    assert metrics.behavior_checks_failed == 1
    assert metrics.behavior_check_pass_rate == pytest.approx(5 / 6)


def test_error_checks_are_excluded_by_status_not_check_name() -> None:
    metrics = calculate_metrics([
        _result("planner.error", EvaluationStatus.ERROR, (False,)),
        EvaluationResult(
            case_id="planner.behavior",
            status=EvaluationStatus.FAILED,
            checks=(EvaluationCheck(name="scenario_execution", passed=False),),
            duration_ms=0,
        ),
    ])
    assert metrics.behavior_checks_total == 1
    assert metrics.behavior_checks_failed == 1


def test_per_dimension_metrics_and_stable_order() -> None:
    results = [
        _result("reliability.one", EvaluationStatus.PASSED, (True,)),
        _result("mcp.one", EvaluationStatus.FAILED, (True, False)),
        _result("rag.one", EvaluationStatus.ERROR, (False,)),
        _result("hitl.one", EvaluationStatus.PASSED, (True,)),
        _result("planner.one", EvaluationStatus.PASSED, (True, True)),
    ]
    metrics = calculate_metrics(results)
    assert tuple(item.dimension for item in metrics.dimensions) == tuple(
        EvaluationDimension
    )
    by_dimension = {item.dimension: item for item in metrics.dimensions}
    assert by_dimension[EvaluationDimension.PLANNER].case_pass_rate == 1.0
    assert by_dimension[EvaluationDimension.HITL].behavior_check_pass_rate == 1.0
    assert by_dimension[EvaluationDimension.RAG].case_errors == 1
    assert by_dimension[EvaluationDimension.RAG].behavior_check_pass_rate is None
    assert by_dimension[EvaluationDimension.MCP].behavior_check_pass_rate == 0.5
    assert by_dimension[EvaluationDimension.RELIABILITY].case_passed == 1


def test_input_order_does_not_change_metrics() -> None:
    results = [
        _result("planner.one", EvaluationStatus.PASSED, (True,)),
        _result("mcp.one", EvaluationStatus.FAILED, (False,)),
    ]
    assert calculate_metrics(results) == calculate_metrics(list(reversed(results)))


def test_empty_results_are_deterministic() -> None:
    metrics = calculate_metrics([])
    assert metrics.case_total == 0
    assert metrics.case_pass_rate == 0.0
    assert metrics.evaluation_error_rate == 0.0
    assert metrics.behavior_check_pass_rate is None
    assert len(metrics.dimensions) == len(EvaluationDimension)
    assert all(item.case_total == 0 for item in metrics.dimensions)


def test_only_error_dimension_has_no_behavior_rate() -> None:
    metrics = calculate_metrics([
        _result("rag.error", EvaluationStatus.ERROR, (False,))
    ])
    rag = next(
        item for item in metrics.dimensions
        if item.dimension is EvaluationDimension.RAG
    )
    assert rag.evaluation_error_rate == 1.0
    assert rag.behavior_checks_total == 0
    assert rag.behavior_check_pass_rate is None


def test_duplicate_case_id_is_rejected() -> None:
    result = _result("planner.same", EvaluationStatus.PASSED, (True,))
    with pytest.raises(EvaluationMetricConfigurationError, match="Duplicate"):
        calculate_metrics([result, result])


def test_unknown_namespace_is_rejected() -> None:
    with pytest.raises(EvaluationMetricConfigurationError, match="unknown"):
        calculate_metrics([
            _result("payments.foo", EvaluationStatus.PASSED, (True,))
        ])


def test_invalid_status_is_rejected() -> None:
    result = _result("planner.invalid", EvaluationStatus.PASSED, (True,))
    object.__setattr__(result, "status", "unexpected")
    with pytest.raises(EvaluationMetricConfigurationError, match="invalid status"):
        calculate_metrics([result])


def test_case_id_without_identifier_is_rejected() -> None:
    with pytest.raises(EvaluationMetricConfigurationError, match="namespace"):
        calculate_metrics([
            _result("planner.", EvaluationStatus.PASSED, (True,))
        ])


def test_metrics_are_immutable_and_rates_are_bounded() -> None:
    metrics = calculate_metrics([
        _result("planner.one", EvaluationStatus.PASSED, (True,))
    ])
    with pytest.raises(FrozenInstanceError):
        metrics.case_total = 4  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metrics.dimensions[0].case_total = 4  # type: ignore[misc]
    rates = [metrics.case_pass_rate, metrics.evaluation_error_rate]
    rates.extend(
        rate
        for item in metrics.dimensions
        for rate in (
            item.case_pass_rate,
            item.evaluation_error_rate,
            item.behavior_check_pass_rate,
        )
        if rate is not None
    )
    assert all(0 <= rate <= 1 for rate in rates)


def test_default_catalog_metrics_are_all_passed() -> None:
    cases = build_default_evaluation_cases()
    results = asyncio.run(EvaluationRunner().run(cases))
    metrics = calculate_metrics(results)
    assert metrics.case_total == len(cases)
    assert metrics.case_failed == 0
    assert metrics.case_errors == 0
    assert metrics.case_pass_rate == 1.0
    assert metrics.behavior_check_pass_rate == 1.0
    assert all(item.case_pass_rate == 1.0 for item in metrics.dimensions)
    assert all(
        item.behavior_check_pass_rate == 1.0 for item in metrics.dimensions
    )
