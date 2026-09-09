import asyncio
from contextlib import asynccontextmanager
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.errors import register_api_exception_handlers
from app.mcp.client import (
    MCPAuthenticationError,
    MCPPermanentDiscoveryError,
    MCPTransientDiscoveryError,
)
from app.mcp.models import MCPRemoteTool, MCPToolResult
from app.mcp.tool_composition import MCPServerClientBinding, compose_mcp_tools
from app.persistence.side_effect_repository import (
    SQLiteSideEffectExecutionRepository,
)
from app.persistence.repository import PersistenceError
from app.providers.types import LLMResponse
from app.reliability.degradation import decide_degradation
from app.reliability.failures import FailureCategory, classify_failure
from app.reliability.retry import (
    OperationSemantics,
    RetryPolicy,
    run_with_retry,
)
from app.reliability.side_effects import (
    SideEffectConflictError,
    SideEffectExecutionStatus,
    SideEffectExecutor,
    SideEffectReplayBlockedError,
)
from app.reliability.timeouts import (
    ReadOperationTimeoutError,
    SideEffectOperationTimeoutError,
    TimeoutPolicy,
    run_with_timeout,
)
from app.services.llm_service import LLMService
from app.services.planner_service import PlannerService, PlanningError
from app.tools.registry import ToolRegistry, create_default_tool_registry


def remote_tool(name="search"):
    return MCPRemoteTool(
        name=name,
        description=f"Remote {name}",
        input_schema={"type": "object", "properties": {}},
    )


class ScriptedMCPClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.list_tools_call_count = 0
        self.call_tool_count = 0

    async def list_tools(self):
        self.list_tools_call_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def call_tool(self, name, arguments):
        self.call_tool_count += 1
        return MCPToolResult(structured_content={"ok": True})


def test_read_only_transient_retry_then_success_has_full_capability():
    client = ScriptedMCPClient(
        [MCPTransientDiscoveryError("private"), [remote_tool()]]
    )
    registry = create_default_tool_registry()
    composition = asyncio.run(
        compose_mcp_tools(
            registry,
            [MCPServerClientBinding("docs", client, required=False)],
        )
    )
    assert client.list_tools_call_count == 2
    assert composition.degradations == ()
    assert registry.contains("mcp_docs_search")
    assert composition.approval_required_actions == {"mcp_docs_search"}


def test_optional_transient_retry_exhaustion_degrades_atomically():
    error = MCPTransientDiscoveryError("token=SUPER_SECRET")
    client = ScriptedMCPClient([error, error])
    registry = create_default_tool_registry()
    composition = asyncio.run(
        compose_mcp_tools(
            registry,
            [MCPServerClientBinding("docs", client, required=False)],
        )
    )
    assert client.list_tools_call_count == 2
    assert registry.contains("get_customer_feedback")
    assert not registry.contains("mcp_docs_search")
    assert composition.approval_required_actions == frozenset()
    assert len(composition.degradations) == 1
    assert "SUPER_SECRET" not in repr(composition.degradations[0])


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (MCPPermanentDiscoveryError("private protocol"), FailureCategory.PERMANENT),
        (MCPAuthenticationError("Bearer SUPER_SECRET"), FailureCategory.CONFIGURATION),
    ],
)
def test_optional_non_transient_discovery_fails_closed_without_retry(
    error, category
):
    client = ScriptedMCPClient([error])
    registry = create_default_tool_registry()
    before = registry.definitions()
    with pytest.raises(type(error)) as raised:
        asyncio.run(
            compose_mcp_tools(
                registry,
                [MCPServerClientBinding("docs", client, required=False)],
            )
        )
    assert classify_failure(raised.value).category is category
    assert client.list_tools_call_count == 1
    assert registry.definitions() == before
    assert "SUPER_SECRET" not in classify_failure(raised.value).safe_message


def test_required_transient_and_later_failure_leave_registry_unchanged():
    successful = ScriptedMCPClient([[remote_tool("available")]])
    optional_error = MCPTransientDiscoveryError("optional unavailable")
    optional = ScriptedMCPClient([optional_error, optional_error])
    required_error = MCPTransientDiscoveryError("required unavailable")
    required = ScriptedMCPClient([required_error, required_error])
    registry = create_default_tool_registry()
    before = registry.definitions()

    with pytest.raises(MCPTransientDiscoveryError):
        asyncio.run(
            compose_mcp_tools(
                registry,
                [
                    MCPServerClientBinding("a", successful),
                    MCPServerClientBinding("b", optional, required=False),
                    MCPServerClientBinding("c", required),
                ],
            )
        )
    assert registry.definitions() == before
    assert successful.list_tools_call_count == 1
    assert optional.list_tools_call_count == 2
    assert required.list_tools_call_count == 2


class PlanProvider:
    model = "test-model"

    async def complete(self, messages, tools=None, tool_choice=None):
        return LLMResponse(
            content=json.dumps(
                {
                    "goal": "Use missing capability",
                    "steps": [
                        {
                            "id": 1,
                            "description": "Use degraded tool",
                            "action": "mcp_docs_search",
                            "arguments": {},
                        }
                    ],
                }
            )
        )


def test_planner_rejects_degraded_action_using_actual_registry():
    error = MCPTransientDiscoveryError("private")
    registry = create_default_tool_registry()
    asyncio.run(
        compose_mcp_tools(
            registry,
            [
                MCPServerClientBinding(
                    "docs", ScriptedMCPClient([error, error]), required=False
                )
            ],
        )
    )
    planner = PlannerService(
        LLMService(PlanProvider()), tool_definitions=registry.definitions()
    )
    with pytest.raises(PlanningError, match="unavailable action"):
        asyncio.run(planner.create_plan("Use missing capability"))


def test_successful_external_action_stays_out_of_reactive_registry():
    registry = create_default_tool_registry()
    composition = asyncio.run(
        compose_mcp_tools(
            registry,
            [MCPServerClientBinding("remote", ScriptedMCPClient([[remote_tool()]]))],
        )
    )
    reactive = registry.excluding(composition.approval_required_actions)
    assert registry.contains("mcp_remote_search")
    assert not reactive.contains("mcp_remote_search")
    assert reactive.contains("get_customer_feedback")


async def initialized_side_effect_executor(tmp_path, timeout=1):
    repository = SQLiteSideEffectExecutionRepository(tmp_path / "ledger.sqlite")
    await repository.initialize()
    return repository, SideEffectExecutor(
        repository, timeout_policy=TimeoutPolicy(timeout)
    )


def test_approved_side_effect_timeout_maps_to_api_409_and_blocks_replay(tmp_path):
    repository, executor = asyncio.run(
        initialized_side_effect_executor(tmp_path, timeout=0.01)
    )
    calls = 0
    app = FastAPI()
    register_api_exception_handlers(app)

    async def remote_operation():
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    @app.post("/approved")
    async def approved():
        return await executor.execute(
            run_id="run-1", step_id=1, action="create_issue",
            arguments={"title": "A"}, operation=remote_operation,
        )

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/approved")
        second = client.post("/approved")
    record = asyncio.run(repository.get("run-1", 1, "create_issue"))
    assert first.status_code == 409
    assert first.json()["error"]["code"] == "side_effect_operation_timeout"
    assert first.json()["error"]["category"] == "ambiguous_side_effect"
    assert "retryable" not in first.text.lower()
    assert "retry-after" not in first.headers
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "side_effect_replay_blocked"
    assert record.status is SideEffectExecutionStatus.AMBIGUOUS
    assert calls == 1


def test_side_effect_cancellation_propagates_and_replay_is_blocked(tmp_path):
    async def scenario():
        repository, executor = await initialized_side_effect_executor(tmp_path)
        started = asyncio.Event()
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            started.set()
            await asyncio.Event().wait()

        kwargs = dict(
            run_id="run-cancel", step_id=1, action="create_issue",
            arguments={}, operation=operation,
        )
        task = asyncio.create_task(executor.execute(**kwargs))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(SideEffectReplayBlockedError):
            await executor.execute(**kwargs)
        return calls, await repository.get("run-cancel", 1, "create_issue")

    calls, record = asyncio.run(scenario())
    assert calls == 1
    assert record.status is SideEffectExecutionStatus.AMBIGUOUS


def test_completed_replay_and_argument_conflict_never_redispatch(tmp_path):
    async def scenario():
        _, executor = await initialized_side_effect_executor(tmp_path)
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"issue_id": 123}

        common = dict(run_id="run-ok", step_id=1, action="create_issue")
        first = await executor.execute(
            **common, arguments={"title": "A"}, operation=operation
        )
        replay = await executor.execute(
            **common, arguments={"title": "A"}, operation=operation
        )
        with pytest.raises(SideEffectConflictError):
            await executor.execute(
                **common, arguments={"title": "B"}, operation=operation
            )
        return first, replay, calls

    first, replay, calls = asyncio.run(scenario())
    assert first == replay == {"issue_id": 123}
    assert calls == 1


def test_completion_persistence_failure_leaves_started_and_blocks_replay(
    tmp_path,
):
    async def scenario():
        repository = SQLiteSideEffectExecutionRepository(
            tmp_path / "completion-failure.sqlite"
        )
        await repository.initialize()

        class CompletionFailingRepository:
            reserve_started = repository.reserve_started
            get = repository.get
            mark_ambiguous = repository.mark_ambiguous

            async def mark_completed(self, **kwargs):
                raise PersistenceError("private persistence detail")

        executor = SideEffectExecutor(CompletionFailingRepository())
        calls = 0

        async def operation():
            nonlocal calls
            calls += 1
            return {"issue_id": 123}

        kwargs = dict(
            run_id="run-persistence", step_id=1, action="create_issue",
            arguments={}, operation=operation,
        )
        with pytest.raises(PersistenceError):
            await executor.execute(**kwargs)
        with pytest.raises(SideEffectReplayBlockedError):
            await executor.execute(**kwargs)
        return calls, await repository.get(
            "run-persistence", 1, "create_issue"
        )

    calls, record = asyncio.run(scenario())
    assert calls == 1
    assert record.status is SideEffectExecutionStatus.STARTED


def test_read_timeout_retry_precedes_optional_degradation():
    attempts = 0

    async def timed_read():
        nonlocal attempts
        attempts += 1
        return await run_with_timeout(
            lambda: asyncio.Event().wait(),
            semantics=OperationSemantics.READ_ONLY,
            policy=TimeoutPolicy(0.005),
        )

    with pytest.raises(ReadOperationTimeoutError) as raised:
        asyncio.run(
            run_with_retry(
                timed_read,
                semantics=OperationSemantics.READ_ONLY,
                policy=RetryPolicy(max_attempts=2),
            )
        )
    failure = classify_failure(raised.value)
    decision = decide_degradation(
        failure, OperationSemantics.READ_ONLY, optional=True
    )
    assert attempts == 2
    assert failure.category is FailureCategory.TRANSIENT
    assert decision.allowed is True


def test_side_effect_timeout_and_unknown_semantics_both_fail_closed():
    side_effect_failure = classify_failure(
        SideEffectOperationTimeoutError("private")
    )
    side_effect_decision = decide_degradation(
        side_effect_failure,
        OperationSemantics.SIDE_EFFECTING,
        optional=True,
    )
    unknown_decision = decide_degradation(
        classify_failure(MCPTransientDiscoveryError("private")),
        OperationSemantics.UNKNOWN,
        optional=True,
    )
    assert side_effect_failure.category is FailureCategory.AMBIGUOUS_SIDE_EFFECT
    assert side_effect_decision.allowed is False
    assert unknown_decision.allowed is False


def test_unified_api_validation_404_405_and_secret_safety():
    class Payload(BaseModel):
        message: str

    app = FastAPI()
    register_api_exception_handlers(app)

    @app.post("/validate")
    async def validate(payload: Payload):
        return payload

    @app.get("/failure")
    async def failure():
        raise RuntimeError(
            "token=SUPER_SECRET password=hunter2 https://user:pass@example.com"
        )

    with TestClient(app, raise_server_exceptions=False) as client:
        invalid = client.post(
            "/validate", json={"message": {"SUPER_SECRET": "hunter2"}}
        )
        missing = client.get("/missing")
        wrong_method = client.get("/validate")
        failed = client.get("/failure")
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "request_validation_error"
    assert "SUPER_SECRET" not in invalid.text
    assert missing.status_code == 404
    assert wrong_method.status_code == 405
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "unclassified_failure"
    assert all(
        secret not in failed.text
        for secret in ("SUPER_SECRET", "hunter2", "user:pass")
    )


def test_optional_mcp_degradation_keeps_healthy_rest_endpoint_available():
    error = MCPTransientDiscoveryError("private unavailable")

    @asynccontextmanager
    async def lifespan(app):
        registry = create_default_tool_registry()
        composition = await compose_mcp_tools(
            registry,
            [
                MCPServerClientBinding(
                    "optional",
                    ScriptedMCPClient([error, error]),
                    required=False,
                )
            ],
        )
        app.state.registry = registry
        app.state.degradations = composition.degradations
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    with TestClient(app) as client:
        response = client.get("/health")
        degradations = app.state.degradations
        registry = app.state.registry
    assert response.status_code == 200
    assert len(degradations) == 1
    assert registry.contains("get_customer_feedback")
    assert not registry.contains("mcp_optional_search")


def test_external_cancellation_never_reaches_generic_api_handler():
    handler_called = False

    async def boundary(operation):
        nonlocal handler_called
        try:
            return await operation()
        except Exception:
            handler_called = True

    async def scenario():
        started = asyncio.Event()

        async def operation():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(boundary(operation))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert handler_called is False
