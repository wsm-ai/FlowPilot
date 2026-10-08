import asyncio
from datetime import datetime, timezone
import json
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from app.observability import AgentEvent, EventType
from app.api.agent import get_planned_agent_service
from app.api.dependencies import get_event_emitter
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.side_effect_repository import SQLiteSideEffectExecutionRepository
from app.providers.types import LLMResponse
from app.reliability.failures import FailureCategory
from app.reliability.side_effects import SideEffectExecutor
from app.services.approval_workflow_service import (
    ApprovalNotPendingError,
    ApprovalWorkflowService,
)
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.registry import ToolRegistry


FIXED_TIME = datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc)


class CapturingEmitter:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[AgentEvent] = []
        self.fail = fail

    def emit(self, event: AgentEvent) -> None:
        if self.fail:
            raise RuntimeError("TOKEN=delivery-secret")
        self.events.append(event)


class Provider:
    model = "observability-fake"

    def __init__(self, payload: dict[str, Any], error: BaseException | None = None) -> None:
        self.payload = payload
        self.error = error

    async def complete(self, **kwargs: Any) -> LLMResponse:
        if self.error is not None:
            raise self.error
        return LLMResponse(content=json.dumps(self.payload), tool_calls=[])


def _read_plan() -> dict[str, Any]:
    return {
        "goal": "Read feedback",
        "steps": [{"id": 1, "description": "Read", "action": "get_customer_feedback", "arguments": {"customer_id": "C001"}, "requires_approval": False}],
    }


def test_planner_success_emits_started_and_completed_without_sensitive_data() -> None:
    emitter = CapturingEmitter()
    planner = PlannerService(
        LLMService(Provider(_read_plan())),
        event_emitter=emitter,
        clock=lambda: FIXED_TIME,
    )
    plan = asyncio.run(planner.create_plan("sensitive customer prompt"))
    assert plan.goal == "Read feedback"
    assert [event.event_type for event in emitter.events] == [
        EventType.PLANNER_STARTED, EventType.PLANNER_COMPLETED,
    ]
    assert all(event.run_id is None and event.thread_id is None for event in emitter.events)
    assert "sensitive customer prompt" not in repr(emitter.events)


def test_planner_failure_emits_safe_category_and_preserves_exception() -> None:
    emitter = CapturingEmitter()
    planner = PlannerService(
        LLMService(Provider({}, PlanningError("TOKEN=provider-secret"))),
        event_emitter=emitter,
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(PlanningError, match="TOKEN=provider-secret"):
        asyncio.run(planner.create_plan("goal"))
    assert [event.event_type for event in emitter.events] == [
        EventType.PLANNER_STARTED, EventType.PLANNER_FAILED,
    ]
    assert emitter.events[-1].failure_category is FailureCategory.VALIDATION
    assert "TOKEN=provider-secret" not in repr(emitter.events)


def test_failure_classification_error_does_not_replace_planner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.reliability.failures as failures

    original_error = PlanningError("original planner failure")

    def fail_classification(exc: BaseException) -> Any:
        raise RuntimeError("classification failed")

    monkeypatch.setattr(failures, "classify_failure", fail_classification)
    planner = PlannerService(LLMService(Provider({}, original_error)))

    with pytest.raises(PlanningError) as captured:
        asyncio.run(planner.create_plan("goal"))

    assert captured.value is original_error


def test_planner_cancellation_propagates() -> None:
    planner = PlannerService(
        LLMService(Provider({}, asyncio.CancelledError()))
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(planner.create_plan("goal"))


def test_logging_delivery_failure_does_not_change_planner_result() -> None:
    planner = PlannerService(
        LLMService(Provider(_read_plan())),
        event_emitter=CapturingEmitter(fail=True),
        clock=lambda: FIXED_TIME,
    )
    assert asyncio.run(planner.create_plan("goal")).goal == "Read feedback"


def test_production_service_composition_injects_operational_emitter() -> None:
    emitter = get_event_emitter()
    service = get_planned_agent_service(
        llm_service=LLMService(Provider(_read_plan())),
        registry=ToolRegistry(),
        approval_required_actions=frozenset(),
        event_emitter=emitter,
    )
    assert service._planner_service._event_emitter is emitter


class SideEffectTool:
    name = "create_issue"
    description = "Create a fake issue"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments: dict[str, Any]) -> dict[str, str]:
        self.calls += 1
        return {"issue_id": "I-1"}


class StaticPlanner:
    async def create_plan(self, goal: str) -> Any:
        from app.schemas.planning import ExecutionPlan
        return ExecutionPlan.model_validate({
            "goal": goal,
            "steps": [{"id": 1, "description": "Create", "action": "create_issue", "arguments": {}, "requires_approval": True}],
        })


class FailingGroundedAnswerService:
    async def synthesize(self, **kwargs: Any) -> Any:
        raise RuntimeError("downstream synthesis bug")


def test_approval_events_follow_real_results_and_preserve_side_effect_count() -> None:
    async def exercise() -> None:
        emitter = CapturingEmitter()
        tool = SideEffectTool()
        registry = ToolRegistry()
        registry.register(tool)
        with TemporaryDirectory(prefix="flowpilot-observability-") as directory:
            repository = SQLiteSideEffectExecutionRepository(f"{directory}/ledger.db")
            await repository.initialize()
            executor = SideEffectExecutor(repository)
            async with async_checkpoint_saver(f"{directory}/checkpoint.db") as saver:
                service = ApprovalWorkflowService(
                    StaticPlanner(), registry, saver,
                    side_effect_executor=executor,
                    event_emitter=emitter,
                    clock=lambda: FIXED_TIME,
                )
                pending = await service.start("Create", thread_id="thread-a")
                assert pending.status == "approval_required"
                assert tool.calls == 0
                completed = await service.resume(
                    "thread-a", "approve", run_id="run-a"
                )
                assert completed.status == "completed"
                assert tool.calls == 1
        assert [event.event_type for event in emitter.events] == [
            EventType.APPROVAL_REQUIRED, EventType.APPROVAL_APPROVED,
        ]
        assert emitter.events[0].run_id is None
        assert emitter.events[0].thread_id == "thread-a"
        assert emitter.events[1].run_id == "run-a"
        assert emitter.events[1].thread_id == "thread-a"

    asyncio.run(exercise())


def test_rejection_event_is_emitted_after_confirmed_rejection_without_dispatch() -> None:
    async def exercise() -> None:
        emitter = CapturingEmitter()
        tool = SideEffectTool()
        registry = ToolRegistry()
        registry.register(tool)
        with TemporaryDirectory(prefix="flowpilot-observability-") as directory:
            async with async_checkpoint_saver(f"{directory}/checkpoint.db") as saver:
                service = ApprovalWorkflowService(
                    StaticPlanner(), registry, saver,
                    event_emitter=emitter, clock=lambda: FIXED_TIME,
                )
                await service.start("Create", thread_id="thread-r")
                rejected = await service.resume(
                    "thread-r", "reject", run_id="run-r"
                )
                assert rejected.status == "rejected"
                assert tool.calls == 0
        assert [event.event_type for event in emitter.events] == [
            EventType.APPROVAL_REQUIRED, EventType.APPROVAL_REJECTED,
        ]

    asyncio.run(exercise())


def test_approved_event_survives_downstream_synthesis_failure_without_redispatch() -> None:
    async def exercise() -> None:
        emitter = CapturingEmitter()
        tool = SideEffectTool()
        registry = ToolRegistry()
        registry.register(tool)
        with TemporaryDirectory(prefix="flowpilot-observability-") as directory:
            repository = SQLiteSideEffectExecutionRepository(f"{directory}/ledger.db")
            await repository.initialize()
            async with async_checkpoint_saver(f"{directory}/checkpoint.db") as saver:
                service = ApprovalWorkflowService(
                    StaticPlanner(), registry, saver,
                    grounded_answer_service=FailingGroundedAnswerService(),
                    side_effect_executor=SideEffectExecutor(repository),
                    event_emitter=emitter, clock=lambda: FIXED_TIME,
                )
                await service.start("Create", thread_id="thread-failure")
                with pytest.raises(RuntimeError, match="downstream synthesis bug"):
                    await service.resume(
                        "thread-failure", "approve", run_id="run-failure"
                    )
                assert tool.calls == 1
                assert [event.event_type for event in emitter.events] == [
                    EventType.APPROVAL_REQUIRED,
                    EventType.APPROVAL_APPROVED,
                ]
                with pytest.raises(ApprovalNotPendingError):
                    await service.resume(
                        "thread-failure", "approve", run_id="run-failure"
                    )
                assert tool.calls == 1

    asyncio.run(exercise())
