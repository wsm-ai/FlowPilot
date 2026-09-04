import asyncio

import pytest
from langgraph.types import Command

from app.graph.approval_nodes import (
    create_approval_node,
    create_prepare_approval_node,
)
from app.graph.approval_workflow import create_approval_graph
from app.graph.execution_nodes import create_approved_plan_step_executor
from app.persistence.checkpoint import async_checkpoint_saver
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.planner_service import PlanningError
from app.tools.base import ToolExecutionError
from app.tools.registry import ToolRegistry, create_default_tool_registry


class SpyIssueTool:
    name = "create_test_issue"
    description = "Create a test issue"
    parameters = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }

    def __init__(self) -> None:
        self.call_count = 0
        self.arguments_received: list[dict[str, object]] = []

    async def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        self.call_count += 1
        self.arguments_received.append(arguments)
        return {"issue_id": "TEST-001", "title": arguments["title"]}


def registry_with_spy() -> tuple[ToolRegistry, SpyIssueTool]:
    registry = create_default_tool_registry()
    tool = SpyIssueTool()
    registry.register(tool)
    return registry, tool


def approval_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Create bug issue",
        steps=[
            PlanStep(
                id=1,
                description="Create a GitHub issue for the bug",
                action="create_test_issue",
                arguments={"title": "Critical login bug"},
                requires_approval=True,
            )
        ],
    )


def initial_state() -> dict[str, object]:
    return {
        "messages": [],
        "llm_response": None,
        "answer": None,
        "executed_tools": [],
        "plan": approval_plan(),
        "current_step_index": 0,
        "route": None,
        "step_results": [],
    }


def config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def expected_payload() -> dict[str, object]:
    return {
        "type": "plan_step_approval",
        "step_id": 1,
        "description": "Create a GitHub issue for the bug",
        "action": "create_test_issue",
        "arguments": {"title": "Critical login bug"},
    }


def test_approval_step_interrupts_with_safe_payload_and_no_side_effects(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "approval.sqlite") as saver:
            registry, tool = registry_with_spy()
            graph = create_approval_graph(registry, saver)
            thread_config = config("approval-thread")
            result = await graph.ainvoke(initial_state(), config=thread_config)
            snapshot = await graph.aget_state(thread_config)
            return result, snapshot, tool

    result, snapshot, tool = asyncio.run(scenario())

    assert result["__interrupt__"][0].value == expected_payload()
    assert snapshot.values["pending_approval"] == expected_payload()
    assert snapshot.values["current_step_index"] == 0
    assert snapshot.values["step_results"] == []
    assert snapshot.values["executed_tools"] == []
    assert snapshot.values.get("approval_decision") is None
    assert snapshot.next == ("approval_node",)
    assert tool.call_count == 0


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_approval_graph_resumes_with_valid_decision(tmp_path, decision: str):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / f"{decision}.sqlite") as saver:
            registry, tool = registry_with_spy()
            graph = create_approval_graph(registry, saver)
            thread_config = config(f"{decision}-thread")
            await graph.ainvoke(initial_state(), config=thread_config)
            result = await graph.ainvoke(
                Command(resume={"decision": decision}),
                config=thread_config,
            )
            return result, tool

    result, tool = asyncio.run(scenario())

    assert result["approval_decision"] == decision
    assert result["pending_approval"] is None
    expected_index = 1 if decision == "approve" else 0
    assert result["route"] == ("end" if decision == "approve" else "reject")
    assert result["current_step_index"] == expected_index
    assert len(result["step_results"]) == expected_index
    assert result["executed_tools"] == []
    assert tool.call_count == expected_index
    if decision == "approve":
        assert tool.arguments_received == [{"title": "Critical login bug"}]
        assert result["step_results"][0]["result"]["issue_id"] == "TEST-001"


@pytest.mark.parametrize(
    "resume_value",
    [{"decision": "maybe"}, "approve", {"other": "approve"}],
)
def test_invalid_resume_payload_raises_planning_error(tmp_path, resume_value):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "invalid.sqlite") as saver:
            registry, _ = registry_with_spy()
            graph = create_approval_graph(registry, saver)
            thread_config = config("invalid-thread")
            await graph.ainvoke(initial_state(), config=thread_config)
            await graph.ainvoke(Command(resume=resume_value), config=thread_config)

    with pytest.raises(PlanningError):
        asyncio.run(scenario())


def test_approval_resume_survives_saver_reconnection(tmp_path):
    database_path = tmp_path / "durable-approval.sqlite"
    thread_config = config("durable-approval")

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver_a:
            registry, tool = registry_with_spy()
            graph_a = create_approval_graph(registry, saver_a)
            await graph_a.ainvoke(initial_state(), config=thread_config)
            snapshot = await graph_a.aget_state(thread_config)
            assert snapshot.values["pending_approval"] == expected_payload()

        async with async_checkpoint_saver(database_path) as saver_b:
            graph_b = create_approval_graph(registry, saver_b)
            result = await graph_b.ainvoke(
                Command(resume={"decision": "approve"}),
                config=thread_config,
            )
            return result, tool

    result, tool = asyncio.run(scenario())

    assert result["approval_decision"] == "approve"
    assert result["pending_approval"] is None
    assert result["current_step_index"] == 1
    assert len(result["step_results"]) == 1
    assert tool.call_count == 1


def test_approval_threads_are_isolated(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "threads.sqlite") as saver:
            registry, tool = registry_with_spy()
            graph = create_approval_graph(registry, saver)
            config_a = config("thread-a")
            config_b = config("thread-b")
            await graph.ainvoke(initial_state(), config=config_a)
            await graph.ainvoke(initial_state(), config=config_b)
            result_a = await graph.ainvoke(
                Command(resume={"decision": "approve"}),
                config=config_a,
            )
            snapshot_b = await graph.aget_state(config_b)
            return result_a, snapshot_b, tool

    result_a, snapshot_b, tool = asyncio.run(scenario())

    assert result_a["approval_decision"] == "approve"
    assert snapshot_b.values.get("approval_decision") is None
    assert snapshot_b.values["pending_approval"] == expected_payload()
    assert snapshot_b.next == ("approval_node",)
    assert snapshot_b.values["step_results"] == []
    assert snapshot_b.values["executed_tools"] == []
    assert tool.call_count == 1


def test_prepare_approval_node_rejects_missing_plan():
    with pytest.raises(PlanningError):
        asyncio.run(create_prepare_approval_node()({}))


def test_prepare_approval_node_rejects_out_of_range_step_index():
    with pytest.raises(PlanningError):
        asyncio.run(
            create_prepare_approval_node()(
                {"plan": approval_plan(), "current_step_index": 1}
            )
        )


def test_prepare_approval_node_rejects_step_without_approval():
    plan = ExecutionPlan(
        goal="Read feedback",
        steps=[
            PlanStep(
                id=1,
                description="Read feedback",
                action="get_customer_feedback",
                requires_approval=False,
            )
        ],
    )

    with pytest.raises(PlanningError):
        asyncio.run(create_prepare_approval_node()({"plan": plan}))


def test_approval_node_requires_pending_approval():
    with pytest.raises(PlanningError):
        asyncio.run(create_approval_node()({}))


def approved_state(
    *,
    plan: ExecutionPlan | None = None,
    index: int = 0,
    decision: str | None = "approve",
    pending_approval=None,
) -> dict[str, object]:
    state = initial_state()
    if plan is None:
        plan = approval_plan()
    state.update(
        {
            "plan": plan,
            "current_step_index": index,
            "approval_decision": decision,
            "pending_approval": pending_approval,
        }
    )
    return state


def test_approved_executor_rejects_missing_plan():
    executor = create_approved_plan_step_executor(ToolRegistry())

    with pytest.raises(PlanningError):
        asyncio.run(executor({"approval_decision": "approve"}))


def test_approved_executor_rejects_invalid_step_index():
    executor = create_approved_plan_step_executor(ToolRegistry())

    with pytest.raises(PlanningError):
        asyncio.run(executor(approved_state(index=1)))


def test_approved_executor_rejects_non_approval_step():
    plan = ExecutionPlan(
        goal="Read feedback",
        steps=[
            PlanStep(
                id=1,
                description="Read feedback",
                action="get_customer_feedback",
            )
        ],
    )
    executor = create_approved_plan_step_executor(ToolRegistry())

    with pytest.raises(PlanningError):
        asyncio.run(executor(approved_state(plan=plan)))


@pytest.mark.parametrize("decision", [None, "reject"])
def test_approved_executor_requires_approve_decision(decision):
    executor = create_approved_plan_step_executor(ToolRegistry())

    with pytest.raises(PlanningError):
        asyncio.run(executor(approved_state(decision=decision)))


def test_approved_executor_rejects_uncleared_pending_approval():
    executor = create_approved_plan_step_executor(ToolRegistry())

    with pytest.raises(PlanningError):
        asyncio.run(executor(approved_state(pending_approval=expected_payload())))


def test_approved_executor_propagates_unknown_tool_error():
    executor = create_approved_plan_step_executor(ToolRegistry())

    with pytest.raises(ToolExecutionError):
        asyncio.run(executor(approved_state()))


def test_approved_executor_executes_registered_tool_once():
    registry, tool = registry_with_spy()
    executor = create_approved_plan_step_executor(registry)

    result = asyncio.run(executor(approved_state()))

    assert tool.call_count == 1
    assert tool.arguments_received == [{"title": "Critical login bug"}]
    assert result["current_step_index"] == 1
    assert result["step_results"][0]["result"]["issue_id"] == "TEST-001"


def test_mixed_plan_runs_normal_step_then_waits_for_approval(tmp_path):
    plan = ExecutionPlan(
        goal="Review feedback and create issue",
        steps=[
            PlanStep(
                id=1,
                description="Read high-priority feedback",
                action="get_customer_feedback",
                arguments={"customer_id": "C001", "priority": "high"},
            ),
            PlanStep(
                id=2,
                description="Create a test issue",
                action="create_test_issue",
                arguments={"title": "Critical login bug"},
                requires_approval=True,
            ),
        ],
    )
    state = initial_state()
    state["plan"] = plan

    async def scenario():
        async with async_checkpoint_saver(tmp_path / "mixed.sqlite") as saver:
            registry, tool = registry_with_spy()
            graph = create_approval_graph(registry, saver)
            thread_config = config("mixed-thread")
            await graph.ainvoke(state, config=thread_config)
            paused = await graph.aget_state(thread_config)
            call_count_before_approval = tool.call_count
            result = await graph.ainvoke(
                Command(resume={"decision": "approve"}),
                config=thread_config,
            )
            return paused, call_count_before_approval, result, tool

    paused, call_count_before_approval, result, tool = asyncio.run(scenario())

    assert paused.values["current_step_index"] == 1
    assert len(paused.values["step_results"]) == 1
    assert call_count_before_approval == 0
    assert tool.call_count == 1
    assert result["current_step_index"] == 2
    assert len(result["step_results"]) == 2


def test_repeated_resume_does_not_repeat_approved_side_effect(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "repeat.sqlite") as saver:
            registry, tool = registry_with_spy()
            graph = create_approval_graph(registry, saver)
            thread_config = config("repeat-thread")
            await graph.ainvoke(initial_state(), config=thread_config)
            await graph.ainvoke(
                Command(resume={"decision": "approve"}),
                config=thread_config,
            )
            await graph.ainvoke(
                Command(resume={"decision": "approve"}),
                config=thread_config,
            )
            return tool.call_count

    assert asyncio.run(scenario()) == 1


def test_approved_tool_failure_does_not_advance_step():
    class FailingTool(SpyIssueTool):
        async def execute(self, arguments: dict[str, object]):
            raise ToolExecutionError("Test tool failed")

    registry = ToolRegistry()
    registry.register(FailingTool())
    executor = create_approved_plan_step_executor(registry)
    state = approved_state()

    with pytest.raises(ToolExecutionError):
        asyncio.run(executor(state))

    assert state["current_step_index"] == 0
    assert state["step_results"] == []
