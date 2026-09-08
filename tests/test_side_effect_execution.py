import asyncio
from dataclasses import FrozenInstanceError

import pytest

from app.graph.execution_nodes import create_approved_plan_step_executor
from app.persistence.repository import PersistenceError
from app.persistence.side_effect_repository import (
    SQLiteSideEffectExecutionRepository,
)
from app.reliability.failures import FailureCategory, FailureDomain, classify_failure
from app.reliability.side_effects import (
    SideEffectConflictError,
    SideEffectExecutionError,
    SideEffectExecutionStatus,
    SideEffectExecutor,
    SideEffectReplayBlockedError,
    digest_arguments,
)
from app.schemas.planning import ExecutionPlan, PlanStep
from app.tools.registry import ToolRegistry


async def initialized_executor(tmp_path):
    repository = SQLiteSideEffectExecutionRepository(tmp_path / "ledger.sqlite")
    await repository.initialize()
    return repository, SideEffectExecutor(repository)


def test_first_execution_is_reserved_then_completed(tmp_path):
    async def scenario():
        repository, executor = await initialized_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            record = await repository.get("run-1", 1, "create_issue")
            assert record.status is SideEffectExecutionStatus.STARTED
            return {"issue_id": 123}

        result = await executor.execute(
            run_id="run-1",
            step_id=1,
            action="create_issue",
            arguments={"title": "A"},
            operation=operation,
        )
        return result, calls, await repository.get("run-1", 1, "create_issue")

    result, calls, record = asyncio.run(scenario())
    assert result == {"issue_id": 123}
    assert calls == 1
    assert record.status is SideEffectExecutionStatus.COMPLETED
    assert record.result == result


def test_completed_duplicate_replays_persisted_result_without_dispatch(tmp_path):
    async def scenario():
        _, executor = await initialized_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"issue_id": 123}

        kwargs = dict(
            run_id="run-1",
            step_id=1,
            action="create_issue",
            arguments={"title": "A"},
            operation=operation,
        )
        first = await executor.execute(**kwargs)
        second = await executor.execute(**kwargs)
        return first, second, calls

    first, second, calls = asyncio.run(scenario())
    assert first == second == {"issue_id": 123}
    assert calls == 1


def test_same_identity_with_other_arguments_conflicts_without_dispatch(tmp_path):
    async def scenario():
        _, executor = await initialized_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"ok": True}

        await executor.execute(
            run_id="run-1", step_id=1, action="create_issue",
            arguments={"title": "A"}, operation=operation,
        )
        with pytest.raises(SideEffectConflictError) as raised:
            await executor.execute(
                run_id="run-1", step_id=1, action="create_issue",
                arguments={"title": "B", "TOKEN": "super-secret"},
                operation=operation,
            )
        return calls, raised.value

    calls, error = asyncio.run(scenario())
    assert calls == 1
    assert str(error) == "Side-effect execution conflicts with existing record"
    assert "super-secret" not in str(error)


@pytest.mark.parametrize("status", ["started", "ambiguous"])
def test_incomplete_execution_blocks_replay(tmp_path, status):
    async def scenario():
        repository, executor = await initialized_executor(tmp_path)
        digest = digest_arguments({"title": "A"})
        await repository.reserve_started(
            run_id="run-1", step_id=1, action="create_issue",
            arguments_digest=digest,
        )
        if status == "ambiguous":
            await repository.mark_ambiguous(
                run_id="run-1", step_id=1, action="create_issue",
                arguments_digest=digest,
            )
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1

        with pytest.raises(SideEffectReplayBlockedError):
            await executor.execute(
                run_id="run-1", step_id=1, action="create_issue",
                arguments={"title": "A"}, operation=operation,
            )
        return calls

    assert asyncio.run(scenario()) == 0


def test_operation_error_marks_ambiguous_and_reraises_original(tmp_path):
    async def scenario():
        repository, executor = await initialized_executor(tmp_path)
        error = RuntimeError("remote uncertain")

        async def operation():
            raise error

        with pytest.raises(RuntimeError) as raised:
            await executor.execute(
                run_id="run-1", step_id=1, action="create_issue",
                arguments={}, operation=operation,
            )
        record = await repository.get("run-1", 1, "create_issue")
        return error, raised.value, record

    error, raised, record = asyncio.run(scenario())
    assert raised is error
    assert record.status is SideEffectExecutionStatus.AMBIGUOUS


def test_cancellation_is_propagated_and_blocks_later_replay(tmp_path):
    async def scenario():
        repository, executor = await initialized_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError()

        kwargs = dict(
            run_id="run-1", step_id=1, action="create_issue",
            arguments={}, operation=operation,
        )
        with pytest.raises(asyncio.CancelledError):
            await executor.execute(**kwargs)
        with pytest.raises(SideEffectReplayBlockedError):
            await executor.execute(**kwargs)
        return calls, await repository.get("run-1", 1, "create_issue")

    calls, record = asyncio.run(scenario())
    assert calls == 1
    assert record.status is SideEffectExecutionStatus.AMBIGUOUS


def test_reservation_failure_prevents_remote_dispatch():
    class FailingRepository:
        async def reserve_started(self, **kwargs):
            raise PersistenceError("private database path")

    calls = 0

    async def operation():
        nonlocal calls
        calls += 1

    executor = SideEffectExecutor(FailingRepository())
    with pytest.raises(PersistenceError):
        asyncio.run(executor.execute(
            run_id="run-1", step_id=1, action="create_issue",
            arguments={}, operation=operation,
        ))
    assert calls == 0


def test_completion_failure_leaves_started_record_and_blocks_replay(tmp_path):
    async def scenario():
        repository, _ = await initialized_executor(tmp_path)

        class CompletionFailingRepository:
            reserve_started = repository.reserve_started
            get = repository.get
            mark_ambiguous = repository.mark_ambiguous

            async def mark_completed(self, **kwargs):
                raise PersistenceError("completion persistence failed")

        executor = SideEffectExecutor(CompletionFailingRepository())
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"issue_id": 123}

        kwargs = dict(
            run_id="run-1", step_id=1, action="create_issue",
            arguments={}, operation=operation,
        )
        with pytest.raises(PersistenceError):
            await executor.execute(**kwargs)
        with pytest.raises(SideEffectReplayBlockedError):
            await executor.execute(**kwargs)
        return calls, await repository.get("run-1", 1, "create_issue")

    calls, record = asyncio.run(scenario())
    assert calls == 1
    assert record.status is SideEffectExecutionStatus.STARTED


def test_concurrent_duplicate_dispatches_at_most_once(tmp_path):
    async def scenario():
        _, executor = await initialized_executor(tmp_path)
        calls = 0
        dispatched = asyncio.Event()
        release = asyncio.Event()

        async def operation():
            nonlocal calls
            calls += 1
            dispatched.set()
            await release.wait()
            return {"ok": True}

        kwargs = dict(
            run_id="run-1", step_id=1, action="create_issue",
            arguments={}, operation=operation,
        )
        first = asyncio.create_task(executor.execute(**kwargs))
        await dispatched.wait()
        second = asyncio.create_task(executor.execute(**kwargs))
        await asyncio.sleep(0)
        release.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        return calls, outcomes

    calls, outcomes = asyncio.run(scenario())
    assert calls == 1
    assert any(outcome == {"ok": True} for outcome in outcomes)
    assert all(
        outcome == {"ok": True}
        or isinstance(outcome, SideEffectReplayBlockedError)
        for outcome in outcomes
    )


def test_different_steps_and_runs_have_independent_identity(tmp_path):
    async def scenario():
        _, executor = await initialized_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return calls

        for run_id, step_id in [("run-a", 1), ("run-a", 2), ("run-b", 1)]:
            await executor.execute(
                run_id=run_id, step_id=step_id, action="create_issue",
                arguments={}, operation=operation,
            )
        return calls

    assert asyncio.run(scenario()) == 3


def test_argument_digest_is_canonical_and_rejects_unsafe_values():
    assert digest_arguments({"a": 1, "b": 2}) == digest_arguments(
        {"b": 2, "a": 1}
    )
    with pytest.raises(SideEffectExecutionError):
        digest_arguments({"value": float("nan")})
    with pytest.raises(SideEffectExecutionError):
        digest_arguments({"value": object()})


def test_invalid_arguments_fail_before_dispatch(tmp_path):
    async def scenario():
        _, executor = await initialized_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1

        with pytest.raises(SideEffectExecutionError):
            await executor.execute(
                run_id="run-1", step_id=1, action="create_issue",
                arguments={"value": float("inf")}, operation=operation,
            )
        return calls

    assert asyncio.run(scenario()) == 0


def test_side_effect_records_and_taxonomy_are_safe_and_immutable(tmp_path):
    async def scenario():
        repository, _ = await initialized_executor(tmp_path)
        reservation = await repository.reserve_started(
            run_id="run-1", step_id=1, action="create_issue",
            arguments_digest=digest_arguments({"TOKEN": "super-secret"}),
        )
        return reservation.record

    record = asyncio.run(scenario())
    with pytest.raises(FrozenInstanceError):
        record.status = SideEffectExecutionStatus.COMPLETED
    replay = classify_failure(SideEffectReplayBlockedError("private"))
    conflict = classify_failure(SideEffectConflictError("private"))
    assert (replay.domain, replay.category) == (
        FailureDomain.WORKFLOW,
        FailureCategory.AMBIGUOUS_SIDE_EFFECT,
    )
    assert (conflict.domain, conflict.category) == (
        FailureDomain.WORKFLOW,
        FailureCategory.PERMANENT,
    )
    assert "super-secret" not in repr(record)


def test_guarded_approved_executor_replays_result_without_second_tool_call(
    tmp_path,
):
    class IssueTool:
        name = "create_issue"
        description = "Create an issue"
        parameters = {"type": "object"}

        def __init__(self):
            self.call_count = 0

        async def execute(self, arguments):
            self.call_count += 1
            return {"issue_id": 123}

    async def scenario():
        _, side_effect_executor = await initialized_executor(tmp_path)
        registry = ToolRegistry()
        tool = IssueTool()
        registry.register(tool)
        executor = create_approved_plan_step_executor(
            registry, side_effect_executor
        )
        state = {
            "plan": ExecutionPlan(
                goal="Create issue",
                steps=[
                    PlanStep(
                        id=7,
                        description="Create issue",
                        action="create_issue",
                        arguments={"title": "Login bug"},
                        requires_approval=True,
                    )
                ],
            ),
            "current_step_index": 0,
            "approval_decision": "approve",
            "pending_approval": None,
            "step_results": [],
        }
        config = {"configurable": {"run_id": "run-7"}}
        first = await executor(state, config)
        second = await executor(state, config)
        return tool.call_count, first, second

    calls, first, second = asyncio.run(scenario())
    assert calls == 1
    assert first["step_results"] == second["step_results"]
    assert first["step_results"][0]["result"] == {"issue_id": 123}


def test_approved_executor_without_protection_fails_closed_before_tool_call():
    class IssueTool:
        name = "create_issue"
        description = "Create an issue"
        parameters = {"type": "object"}

        def __init__(self):
            self.call_count = 0

        async def execute(self, arguments):
            self.call_count += 1

    registry = ToolRegistry()
    tool = IssueTool()
    registry.register(tool)
    executor = create_approved_plan_step_executor(registry)
    state = {
        "plan": ExecutionPlan(
            goal="Create issue",
            steps=[
                PlanStep(
                    id=1,
                    description="Create issue",
                    action="create_issue",
                    requires_approval=True,
                )
            ],
        ),
        "current_step_index": 0,
        "approval_decision": "approve",
        "pending_approval": None,
    }

    with pytest.raises(
        SideEffectExecutionError, match="protection is required"
    ):
        asyncio.run(executor(state))
    assert tool.call_count == 0


@pytest.mark.parametrize("base_error", [KeyboardInterrupt(), SystemExit(7)])
def test_process_control_base_exceptions_propagate_and_started_blocks_replay(
    tmp_path, base_error
):
    async def scenario():
        repository, executor = await initialized_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            raise base_error

        kwargs = dict(
            run_id="run-control",
            step_id=1,
            action="create_issue",
            arguments={},
            operation=operation,
        )
        with pytest.raises(type(base_error)) as raised:
            await executor.execute(**kwargs)
        with pytest.raises(SideEffectReplayBlockedError):
            await executor.execute(**kwargs)
        return raised.value, calls, await repository.get(
            "run-control", 1, "create_issue"
        )

    raised, calls, record = asyncio.run(scenario())
    assert raised is base_error
    assert calls == 1
    assert record.status is SideEffectExecutionStatus.STARTED
