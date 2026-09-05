import asyncio

import pytest

from app.grounding.lifecycle import attempt_grounded_synthesis
from app.schemas.planning import ExecutionPlan, PlanStep


def execution_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Review customer feedback",
        steps=[
            PlanStep(
                id=1,
                description="Read customer feedback",
                action="get_customer_feedback",
                arguments={"customer_id": "C001"},
                requires_approval=False,
            )
        ],
    )


def test_attempt_grounded_synthesis_does_not_swallow_unexpected_errors():
    class FakeSynthesizer:
        async def synthesize(self, *, goal, plan, step_results):
            raise RuntimeError("unexpected bug")

    with pytest.raises(RuntimeError, match="unexpected bug"):
        asyncio.run(
            attempt_grounded_synthesis(
                FakeSynthesizer(),
                goal="Review customer feedback",
                plan=execution_plan(),
                step_results=[],
            )
        )


def test_attempt_grounded_synthesis_without_service_is_not_attempted():
    outcome = asyncio.run(
        attempt_grounded_synthesis(
            None,
            goal="Review customer feedback",
            plan=execution_plan(),
            step_results=[],
        )
    )

    assert outcome.status == "not_attempted"
    assert outcome.grounded_answer is None
    assert outcome.error_type is None
