import asyncio

import pytest
from langgraph.types import Command

from app.graph.approval_nodes import (
    create_approval_node,
    create_prepare_approval_node,
)
from app.graph.approval_workflow import create_approval_graph
from app.persistence.checkpoint import async_checkpoint_saver
from app.schemas.planning import ExecutionPlan, PlanStep
from app.services.planner_service import PlanningError


def approval_plan() -> ExecutionPlan:
    return ExecutionPlan(
        goal="Create bug issue",
        steps=[
            PlanStep(
                id=1,
                description="Create a GitHub issue for the bug",
                action="create_github_issue",
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
        "action": "create_github_issue",
        "arguments": {"title": "Critical login bug"},
    }


def test_approval_step_interrupts_with_safe_payload_and_no_side_effects(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "approval.sqlite") as saver:
            graph = create_approval_graph(saver)
            thread_config = config("approval-thread")
            result = await graph.ainvoke(initial_state(), config=thread_config)
            snapshot = await graph.aget_state(thread_config)
            return result, snapshot

    result, snapshot = asyncio.run(scenario())

    assert result["__interrupt__"][0].value == expected_payload()
    assert snapshot.values["pending_approval"] == expected_payload()
    assert snapshot.values["current_step_index"] == 0
    assert snapshot.values["step_results"] == []
    assert snapshot.values["executed_tools"] == []
    assert snapshot.values.get("approval_decision") is None
    assert snapshot.next == ("approval_node",)


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_approval_graph_resumes_with_valid_decision(tmp_path, decision: str):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / f"{decision}.sqlite") as saver:
            graph = create_approval_graph(saver)
            thread_config = config(f"{decision}-thread")
            await graph.ainvoke(initial_state(), config=thread_config)
            return await graph.ainvoke(
                Command(resume={"decision": decision}),
                config=thread_config,
            )

    result = asyncio.run(scenario())

    assert result["approval_decision"] == decision
    assert result["pending_approval"] is None
    assert result["route"] == decision
    assert result["current_step_index"] == 0
    assert result["step_results"] == []
    assert result["executed_tools"] == []


@pytest.mark.parametrize(
    "resume_value",
    [{"decision": "maybe"}, "approve", {"other": "approve"}],
)
def test_invalid_resume_payload_raises_planning_error(tmp_path, resume_value):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "invalid.sqlite") as saver:
            graph = create_approval_graph(saver)
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
            graph_a = create_approval_graph(saver_a)
            await graph_a.ainvoke(initial_state(), config=thread_config)
            snapshot = await graph_a.aget_state(thread_config)
            assert snapshot.values["pending_approval"] == expected_payload()

        async with async_checkpoint_saver(database_path) as saver_b:
            graph_b = create_approval_graph(saver_b)
            return await graph_b.ainvoke(
                Command(resume={"decision": "approve"}),
                config=thread_config,
            )

    result = asyncio.run(scenario())

    assert result["approval_decision"] == "approve"
    assert result["pending_approval"] is None


def test_approval_threads_are_isolated(tmp_path):
    async def scenario():
        async with async_checkpoint_saver(tmp_path / "threads.sqlite") as saver:
            graph = create_approval_graph(saver)
            config_a = config("thread-a")
            config_b = config("thread-b")
            await graph.ainvoke(initial_state(), config=config_a)
            await graph.ainvoke(initial_state(), config=config_b)
            result_a = await graph.ainvoke(
                Command(resume={"decision": "approve"}),
                config=config_a,
            )
            snapshot_b = await graph.aget_state(config_b)
            return result_a, snapshot_b

    result_a, snapshot_b = asyncio.run(scenario())

    assert result_a["approval_decision"] == "approve"
    assert snapshot_b.values.get("approval_decision") is None
    assert snapshot_b.values["pending_approval"] == expected_payload()
    assert snapshot_b.next == ("approval_node",)
    assert snapshot_b.values["step_results"] == []
    assert snapshot_b.values["executed_tools"] == []


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
