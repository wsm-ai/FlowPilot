import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

import app.api.agent as agent_api
import app.main as application_main
from app.api.dependencies import (
    get_checkpointer,
    get_event_emitter,
    get_llm_service,
    get_mcp_approval_required_actions,
    get_run_repository,
    get_side_effect_executor,
    get_tool_registry,
    get_trace_emitter,
)
from app.observability import (
    EventType,
    NoOpAsyncTraceEmitter,
    NoOpEventEmitter,
    SQLiteTraceEmitter,
)
from app.persistence.checkpoint import async_checkpoint_saver
from app.persistence.side_effect_repository import SQLiteSideEffectExecutionRepository
from app.persistence.sqlite_repository import SQLiteRunRepository
from app.persistence.trace_repository import SQLiteTraceRepository
from app.providers.types import LLMResponse
from app.reliability.side_effects import SideEffectExecutor
from app.services.approval_workflow_service import (
    ApprovalNotPendingError,
    ApprovalThreadConflictError,
    ApprovalThreadNotFoundError,
    ApprovalWorkflowService,
)
from app.services.planned_agent_service import PlannedAgentService
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.registry import ToolRegistry
from app.security.tool_authorization import ToolRisk


class Provider:
    model = "trace-integration-fake"

    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.error = error

    async def complete(self, **kwargs: Any) -> LLMResponse:
        if self.error is not None:
            raise self.error
        return LLMResponse(
            content=json.dumps(self.payload),
            tool_calls=[],
        )


class SideEffectTool:
    name = "create_issue"
    description = "Create a test-only issue"
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, arguments: dict[str, Any]) -> dict[str, str]:
        self.calls += 1
        return {"issue_id": "I-1"}


def _plan() -> dict[str, Any]:
    return {
        "goal": "Create issue",
        "steps": [{
            "id": 1,
            "description": "Create issue",
            "action": "create_issue",
            "arguments": {},
            "requires_approval": False,
        }],
    }


async def _components(path: Path, *, trace_id: str = "trace-fixed"):
    trace_repository = SQLiteTraceRepository(path / "application.db")
    await trace_repository.initialize()
    trace_emitter = SQLiteTraceEmitter(trace_repository)
    side_repository = SQLiteSideEffectExecutionRepository(
        path / "application.db"
    )
    await side_repository.initialize()
    tool = SideEffectTool()
    registry = ToolRegistry()
    registry.register(tool, risk=ToolRisk.READ_ONLY)
    planner = PlannerService(
        LLMService(Provider(_plan())),
        registry.definitions(),
        {tool.name},
        trace_emitter=trace_emitter,
    )
    return (
        trace_repository,
        trace_emitter,
        SideEffectExecutor(side_repository),
        tool,
        registry,
        planner,
        lambda: trace_id,
    )


def test_production_dependencies_supply_initialized_trace_emitter(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(trace_emitter=emitter))
        )
        assert get_trace_emitter(request) is emitter

    asyncio.run(exercise())


def test_planner_success_and_failure_are_persisted_safely(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)
        success = PlannerService(
            LLMService(Provider(_plan())), trace_emitter=emitter
        )
        await success.create_plan(
            "TOKEN=goal-must-not-persist",
            trace_id="trace-success",
            thread_id="thread-success",
        )
        failure_error = PlanningError("TOKEN=error-must-not-persist")
        failure = PlannerService(
            LLMService(Provider(error=failure_error)),
            trace_emitter=emitter,
        )
        with pytest.raises(PlanningError) as captured:
            await failure.create_plan(
                "secret goal", trace_id="trace-failure"
            )
        assert captured.value is failure_error
        success_trace = await repository.get_trace("trace-success")
        failure_trace = await repository.get_trace("trace-failure")
        assert [item.event_type for item in success_trace.records] == [
            EventType.PLANNER_STARTED,
            EventType.PLANNER_COMPLETED,
        ]
        assert [item.event_type for item in failure_trace.records] == [
            EventType.PLANNER_STARTED,
            EventType.PLANNER_FAILED,
        ]
        raw = (tmp_path / "trace.db").read_bytes()
        assert b"TOKEN=" not in raw and b"secret goal" not in raw

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("decision", "final_event", "expected_calls"),
    [
        ("approve", EventType.APPROVAL_APPROVED, 1),
        ("reject", EventType.APPROVAL_REJECTED, 0),
    ],
)
def test_hitl_start_and_resume_share_durable_trace(
    tmp_path: Path,
    decision: str,
    final_event: EventType,
    expected_calls: int,
) -> None:
    async def exercise() -> None:
        (
            repository, emitter, executor, tool, registry, planner, factory,
        ) = await _components(tmp_path)
        checkpoint = tmp_path / "checkpoint.db"
        async with async_checkpoint_saver(checkpoint) as saver:
            service = ApprovalWorkflowService(
                planner, registry, saver,
                side_effect_executor=executor,
                trace_emitter=emitter,
                trace_id_factory=factory,
            )
            pending = await service.start("Create", thread_id="thread-a")
            assert pending.status == "approval_required" and tool.calls == 0

        restarted_repository = SQLiteTraceRepository(tmp_path / "application.db")
        await restarted_repository.initialize()
        restarted_emitter = SQLiteTraceEmitter(restarted_repository)
        async with async_checkpoint_saver(checkpoint) as saver:
            restarted = ApprovalWorkflowService(
                planner, registry, saver,
                side_effect_executor=executor,
                trace_emitter=restarted_emitter,
                trace_id_factory=lambda: "must-not-be-used",
            )
            result = await restarted.resume(
                "thread-a", decision, run_id="run-a"  # type: ignore[arg-type]
            )
            assert result.status == (
                "completed" if decision == "approve" else "rejected"
            )
            with pytest.raises(ApprovalNotPendingError):
                await restarted.resume(
                    "thread-a", decision, run_id="run-a"  # type: ignore[arg-type]
                )
        assert tool.calls == expected_calls
        assert await restarted_repository.get_trace_id_for_thread(
            "thread-a"
        ) == "trace-fixed"
        trace = await restarted_repository.get_trace("trace-fixed")
        assert [record.event_type for record in trace.records] == [
            EventType.PLANNER_STARTED,
            EventType.PLANNER_COMPLETED,
            EventType.APPROVAL_REQUIRED,
            final_event,
        ]
        assert trace.records[-1].run_id == "run-a"
        assert trace.records[-1].thread_id == "thread-a"
        assert all(record.run_id != record.trace_id for record in trace.records)

    asyncio.run(exercise())


def test_different_threads_do_not_share_trace_and_invalid_resume_adds_nothing(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        values = iter(("trace-a", "trace-b"))
        repository, emitter, executor, _, registry, planner, _ = await _components(
            tmp_path
        )
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner, registry, saver,
                side_effect_executor=executor,
                trace_emitter=emitter,
                trace_id_factory=lambda: next(values),
            )
            await asyncio.gather(
                service.start("Create A", thread_id="thread-a"),
                service.start("Create B", thread_id="thread-b"),
            )
            with pytest.raises(ApprovalThreadNotFoundError):
                await service.resume("missing", "approve", run_id="run-x")
        assert await repository.get_trace_id_for_thread("thread-a") == "trace-a"
        assert await repository.get_trace_id_for_thread("thread-b") == "trace-b"
        assert await repository.get_trace_id_for_thread("missing") is None

    asyncio.run(exercise())


class FailingTraceRepository:
    async def append_event(self, trace_id: str, event: Any) -> Any:
        raise RuntimeError("TOKEN=trace-write-failure")

    async def associate_thread(self, thread_id: str, trace_id: str) -> None:
        raise RuntimeError("association failed")

    async def get_trace_id_for_thread(self, thread_id: str) -> str | None:
        raise RuntimeError("lookup failed")


def test_trace_failure_does_not_change_business_or_retry_side_effect(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        _, _, executor, tool, registry, _, factory = await _components(tmp_path)
        emitter = SQLiteTraceEmitter(FailingTraceRepository())
        planner = PlannerService(
            LLMService(Provider(_plan())),
            registry.definitions(),
            {tool.name},
            trace_emitter=emitter,
        )
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner, registry, saver,
                side_effect_executor=executor,
                trace_emitter=emitter,
                trace_id_factory=factory,
            )
            pending = await service.start("Create", thread_id="thread-a")
            assert pending.status == "approval_required"
            completed = await service.resume(
                "thread-a", "approve", run_id="run-a"
            )
            assert completed.status == "completed"
            assert tool.calls == 1

    asyncio.run(exercise())


class FailingSynthesis:
    async def synthesize(self, **kwargs: Any) -> Any:
        raise RuntimeError("downstream synthesis failed")


def test_confirmed_approval_is_persisted_before_synthesis_failure(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository, emitter, executor, tool, registry, planner, factory = (
            await _components(tmp_path)
        )
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner, registry, saver,
                grounded_answer_service=FailingSynthesis(),
                side_effect_executor=executor,
                trace_emitter=emitter,
                trace_id_factory=factory,
            )
            await service.start("Create", thread_id="thread-a")
            with pytest.raises(RuntimeError, match="downstream synthesis failed"):
                await service.resume("thread-a", "approve", run_id="run-a")
        trace = await repository.get_trace("trace-fixed")
        assert trace.records[-1].event_type is EventType.APPROVAL_APPROVED
        assert tool.calls == 1

    asyncio.run(exercise())


def test_no_background_task_is_created_for_trace_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("background tasks are forbidden")

        monkeypatch.setattr(asyncio, "create_task", forbidden)
        planner = PlannerService(
            LLMService(Provider(_plan())), trace_emitter=emitter
        )
        await planner.create_plan("Create", trace_id="trace-a")
        assert len((await repository.get_trace("trace-a")).records) == 2

    asyncio.run(exercise())


def test_planned_agent_requests_receive_distinct_persisted_traces(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "planned.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)
        registry = ToolRegistry()
        registry.register(SideEffectTool(), risk=ToolRisk.READ_ONLY)
        planner = PlannerService(
            LLMService(Provider(_plan())),
            trace_emitter=emitter,
        )
        trace_ids = iter(("planned-trace-a", "planned-trace-b"))
        service = PlannedAgentService(
            planner,
            registry,
            trace_id_factory=lambda: next(trace_ids),
        )

        first, second = await asyncio.gather(
            service.run("TOKEN=first"), service.run("TOKEN=second")
        )

        assert first.status == second.status == "completed"
        for trace_id in ("planned-trace-a", "planned-trace-b"):
            trace = await repository.get_trace(trace_id)
            assert [record.event_type for record in trace.records] == [
                EventType.PLANNER_STARTED,
                EventType.PLANNER_COMPLETED,
            ]
            assert all(record.run_id is None for record in trace.records)
            assert all(record.thread_id is None for record in trace.records)
        assert b"TOKEN=" not in (tmp_path / "planned.db").read_bytes()

    asyncio.run(exercise())


def test_concurrent_start_same_thread_has_one_durable_owner(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository, emitter, executor, tool, registry, planner, _ = (
            await _components(tmp_path)
        )
        trace_ids = iter(("concurrent-trace-a", "concurrent-trace-b"))
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner,
                registry,
                saver,
                side_effect_executor=executor,
                trace_emitter=emitter,
                trace_id_factory=lambda: next(trace_ids),
            )
            outcomes = await asyncio.gather(
                service.start("Create A", thread_id="shared-thread"),
                service.start("Create B", thread_id="shared-thread"),
                return_exceptions=True,
            )
            assert sum(
                isinstance(item, ApprovalThreadConflictError)
                for item in outcomes
            ) == 1
            assert sum(not isinstance(item, Exception) for item in outcomes) == 1
            owner = await repository.get_trace_id_for_thread("shared-thread")
            assert owner in {"concurrent-trace-a", "concurrent-trace-b"}
            completed = await service.resume(
                "shared-thread", "approve", run_id="run-shared"
            )
            assert completed.status == "completed"
        assert tool.calls == 1

    asyncio.run(exercise())


def test_trace_initialization_failure_degrades_to_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_initialize(self: Any) -> None:
        raise RuntimeError("TOKEN=database failure")

    async def exercise() -> None:
        monkeypatch.setattr(
            application_main.SQLiteTraceRepository,
            "initialize",
            fail_initialize,
        )
        repository, emitter = await application_main.initialize_trace_persistence(
            "unused.db"
        )
        assert repository is None
        assert isinstance(emitter, NoOpAsyncTraceEmitter)
        await emitter.emit("trace-a", object())  # type: ignore[arg-type]

    asyncio.run(exercise())


def test_planner_failure_releases_reservation_before_checkpoint(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)
        planner = PlannerService(
            LLMService(Provider(error=PlanningError("planner failed"))),
            trace_emitter=emitter,
        )
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner,
                ToolRegistry(),
                saver,
                trace_emitter=emitter,
                trace_id_factory=lambda: "planner-failure-trace",
            )
            with pytest.raises(PlanningError, match="planner failed"):
                await service.start("Create", thread_id="planner-failure-thread")
        assert await repository.get_trace_id_for_thread(
            "planner-failure-thread"
        ) is None

    asyncio.run(exercise())


def test_failure_after_checkpoint_keeps_durable_trace_identity(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository, emitter, executor, tool, registry, _, _ = await _components(
            tmp_path
        )
        planner = PlannerService(
            LLMService(Provider(_plan())),
            registry.definitions(),
            trace_emitter=emitter,
        )
        checkpoint = tmp_path / "checkpoint.db"
        async with async_checkpoint_saver(checkpoint) as saver:
            service = ApprovalWorkflowService(
                planner,
                registry,
                saver,
                grounded_answer_service=FailingSynthesis(),
                side_effect_executor=executor,
                trace_emitter=emitter,
                trace_id_factory=lambda: "post-checkpoint-trace",
            )
            with pytest.raises(RuntimeError, match="downstream synthesis failed"):
                await service.start("Create", thread_id="post-checkpoint-thread")
        restarted = SQLiteTraceRepository(tmp_path / "application.db")
        await restarted.initialize()
        assert await restarted.get_trace_id_for_thread(
            "post-checkpoint-thread"
        ) == "post-checkpoint-trace"
        assert tool.calls == 1

    asyncio.run(exercise())


def test_unknown_checkpoint_state_keeps_reservation(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)
        planner = PlannerService(
            LLMService(Provider(error=PlanningError("original failure"))),
            trace_emitter=emitter,
        )
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner,
                ToolRegistry(),
                saver,
                trace_emitter=emitter,
                trace_id_factory=lambda: "unknown-state-trace",
            )
            original_get_state = service._graph.aget_state
            calls = 0

            async def unreliable_get_state(config: Any) -> Any:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return await original_get_state(config)
                raise RuntimeError("checkpoint unavailable")

            service._graph.aget_state = unreliable_get_state
            with pytest.raises(PlanningError, match="original failure"):
                await service.start("Create", thread_id="unknown-state-thread")
        assert await repository.get_trace_id_for_thread(
            "unknown-state-thread"
        ) == "unknown-state-trace"

    asyncio.run(exercise())


def test_cancellation_during_planner_propagates_and_safely_releases(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        entered = asyncio.Event()

        class BlockingProvider:
            model = "blocking"

            async def complete(self, **kwargs: Any) -> LLMResponse:
                entered.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        repository = SQLiteTraceRepository(tmp_path / "trace.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)
        planner = PlannerService(LLMService(BlockingProvider()), trace_emitter=emitter)
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner,
                ToolRegistry(),
                saver,
                trace_emitter=emitter,
                trace_id_factory=lambda: "cancelled-trace",
            )
            task = asyncio.create_task(
                service.start("Create", thread_id="cancelled-thread")
            )
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert await repository.get_trace_id_for_thread("cancelled-thread") is None

    asyncio.run(exercise())


def test_release_failure_does_not_replace_original_failure(tmp_path: Path) -> None:
    async def exercise() -> None:
        class ReleaseFailingRepository(SQLiteTraceRepository):
            async def release_thread(self, thread_id: str, trace_id: str) -> None:
                raise RuntimeError("release failed")

        repository = ReleaseFailingRepository(tmp_path / "trace.db")
        await repository.initialize()
        emitter = SQLiteTraceEmitter(repository)
        original = PlanningError("original planner failure")
        planner = PlannerService(
            LLMService(Provider(error=original)), trace_emitter=emitter
        )
        async with async_checkpoint_saver(tmp_path / "checkpoint.db") as saver:
            service = ApprovalWorkflowService(
                planner,
                ToolRegistry(),
                saver,
                trace_emitter=emitter,
                trace_id_factory=lambda: "release-failure-trace",
            )
            with pytest.raises(PlanningError) as captured:
                await service.start("Create", thread_id="release-failure-thread")
            assert captured.value is original

    asyncio.run(exercise())


def test_real_plan_run_route_persists_trace_with_production_dependencies(
    tmp_path: Path,
) -> None:
    database = tmp_path / "application.db"
    checkpoint = tmp_path / "checkpoint.db"
    run_repository = SQLiteRunRepository(database)
    trace_repository = SQLiteTraceRepository(database)
    side_repository = SQLiteSideEffectExecutionRepository(database)
    registry = ToolRegistry()
    tool = SideEffectTool()
    registry.register(tool)
    emitter = SQLiteTraceEmitter(trace_repository)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_repository.initialize()
        await trace_repository.initialize()
        await side_repository.initialize()
        async with async_checkpoint_saver(checkpoint) as saver:
            app.state.checkpointer = saver
            yield

    test_app = FastAPI(lifespan=lifespan)
    test_app.include_router(agent_api.router)
    test_app.dependency_overrides[get_llm_service] = lambda: LLMService(
        Provider(_plan())
    )
    test_app.dependency_overrides[get_tool_registry] = lambda: registry
    test_app.dependency_overrides[get_mcp_approval_required_actions] = (
        lambda: frozenset({tool.name})
    )
    test_app.dependency_overrides[get_event_emitter] = NoOpEventEmitter
    test_app.dependency_overrides[get_trace_emitter] = lambda: emitter
    test_app.dependency_overrides[get_run_repository] = lambda: run_repository
    test_app.dependency_overrides[get_side_effect_executor] = lambda: SideEffectExecutor(
        side_repository
    )

    def checkpoint_dependency(request: Request) -> Any:
        return request.app.state.checkpointer

    test_app.dependency_overrides[get_checkpointer] = checkpoint_dependency

    with TestClient(test_app) as client:
        response = client.post(
            "/api/v1/agent/plan-run",
            json={"goal": "Create issue", "thread_id": "http-thread"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "approval_required"
    trace_id = asyncio.run(
        trace_repository.get_trace_id_for_thread("http-thread")
    )
    assert trace_id is not None
    trace = asyncio.run(trace_repository.get_trace(trace_id))
    assert [record.event_type for record in trace.records] == [
        EventType.PLANNER_STARTED,
        EventType.PLANNER_COMPLETED,
        EventType.APPROVAL_REQUIRED,
    ]
    assert tool.calls == 0
