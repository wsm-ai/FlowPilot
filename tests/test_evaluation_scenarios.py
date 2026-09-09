import asyncio

import pytest

import app.evaluation.scenarios as scenarios
from app.evaluation import (
    EvaluationRunner,
    EvaluationStatus,
    build_default_evaluation_cases,
)
from app.mcp.tool_composition import MCPToolComposition
from app.schemas.planning import ExecutionPlan
from app.services.planner_service import PlanningError


EXPECTED_CASE_IDS = (
    "planner.read_customer_feedback",
    "planner.reject_unknown_action",
    "hitl.requires_approval",
    "hitl.reject_side_effect",
    "hitl.approve_side_effect",
    "rag.grounded_citation",
    "rag.insufficient_evidence",
    "mcp.reactive_least_privilege",
    "mcp.degraded_capability",
    "reliability.ambiguous_side_effect",
)


def test_default_catalog_has_unique_stable_order() -> None:
    ids = tuple(case.case_id for case in build_default_evaluation_cases())
    assert ids == EXPECTED_CASE_IDS
    assert len(ids) == len(set(ids))


def test_default_scenarios_return_non_empty_checks() -> None:
    async def exercise() -> None:
        for case in build_default_evaluation_cases():
            assert await case.scenario.run()

    asyncio.run(exercise())


@pytest.mark.parametrize("case_id", EXPECTED_CASE_IDS)
def test_default_agent_behavior_scenario_passes(case_id: str) -> None:
    case = next(
        case for case in build_default_evaluation_cases() if case.case_id == case_id
    )
    result = asyncio.run(EvaluationRunner().run([case]))[0]
    assert result.status is EvaluationStatus.PASSED
    assert result.checks
    assert all(check.passed for check in result.checks)


def test_approval_rejection_records_zero_dispatch_behavior() -> None:
    case = build_default_evaluation_cases()[3]
    result = asyncio.run(EvaluationRunner().run([case]))[0]
    checks = {check.name: check.passed for check in result.checks}
    assert checks["side_effect_not_executed"] is True


def test_approval_acceptance_records_single_dispatch_behavior() -> None:
    case = build_default_evaluation_cases()[4]
    result = asyncio.run(EvaluationRunner().run([case]))[0]
    checks = {check.name: check.passed for check in result.checks}
    assert checks["single_dispatch"] is True


def test_default_catalog_runner_is_all_passed_and_ordered() -> None:
    results = asyncio.run(
        EvaluationRunner().run(build_default_evaluation_cases())
    )
    assert tuple(result.case_id for result in results) == EXPECTED_CASE_IDS
    assert all(result.status is EvaluationStatus.PASSED for result in results)
    assert all(result.status is not EvaluationStatus.ERROR for result in results)


def test_planner_missing_step_is_failed_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def return_empty_plan(*args: object, **kwargs: object) -> object:
        return type("EmptyPlan", (), {"steps": []})()

    monkeypatch.setattr(scenarios.PlannerService, "create_plan", return_empty_plan)
    result = asyncio.run(EvaluationRunner().run([build_default_evaluation_cases()[0]]))[0]
    assert result.status is EvaluationStatus.FAILED
    assert all(not check.passed for check in result.checks)


def test_mcp_missing_expected_tool_is_failed_not_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def compose_without_tools(*args: object, **kwargs: object) -> MCPToolComposition:
        return MCPToolComposition(
            tools=(), local_names=(), approval_required_actions=frozenset()
        )

    monkeypatch.setattr(scenarios, "compose_mcp_tools", compose_without_tools)
    result = asyncio.run(EvaluationRunner().run([build_default_evaluation_cases()[7]]))[0]
    assert result.status is EvaluationStatus.FAILED
    assert all(not check.passed for check in result.checks)


@pytest.mark.parametrize("case_index", [1, 8])
def test_unrelated_planning_error_is_not_false_green(
    monkeypatch: pytest.MonkeyPatch,
    case_index: int,
) -> None:
    async def raise_unrelated_error(*args: object, **kwargs: object) -> ExecutionPlan:
        raise PlanningError("Unrelated deterministic planner failure")

    monkeypatch.setattr(scenarios.PlannerService, "create_plan", raise_unrelated_error)
    result = asyncio.run(
        EvaluationRunner().run([build_default_evaluation_cases()[case_index]])
    )[0]
    assert result.status is EvaluationStatus.ERROR
    assert not result.checks[0].passed
