import asyncio

import pytest

from app.graph.execution_workflow import create_plan_execution_graph
from app.persistence.checkpoint import async_checkpoint_saver
from app.schemas.planning import ExecutionPlan, PlanStep
from app.tools.registry import create_default_tool_registry


def create_plan(customer_id: str) -> ExecutionPlan:
    return ExecutionPlan(
        goal=f"Review feedback for {customer_id}",
        steps=[
            PlanStep(
                id=1,
                description=f"Retrieve high-priority feedback for {customer_id}",
                action="get_customer_feedback",
                arguments={"customer_id": customer_id, "priority": "high"},
            )
        ],
    )


def initial_state(customer_id: str):
    plan = create_plan(customer_id)
    return {
        "messages": [],
        "llm_response": None,
        "answer": None,
        "executed_tools": [],
        "goal": plan.goal,
        "plan": plan,
        "current_step_index": 0,
        "route": None,
        "step_results": [],
    }


def config(thread_id: str):
    return {"configurable": {"thread_id": thread_id}}


def test_checkpoint_database_and_final_state_are_saved(tmp_path):
    database_path = tmp_path / "checkpoints.sqlite"

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver:
            graph = create_plan_execution_graph(
                create_default_tool_registry(),
                checkpointer=saver,
            )
            thread_config = config("thread-001")
            await graph.ainvoke(initial_state("C001"), config=thread_config)
            snapshot = await graph.aget_state(thread_config)
            history = [
                item async for item in graph.aget_state_history(thread_config)
            ]
            return snapshot, history

    snapshot, history = asyncio.run(scenario())

    assert database_path.exists()
    assert snapshot.values["current_step_index"] == 1
    assert snapshot.values["route"] == "end"
    assert len(snapshot.values["step_results"]) == 1
    records = snapshot.values["step_results"][0]["result"]
    assert {record["id"] for record in records} == {"FB-001", "FB-003"}
    assert len(history) >= 1


def test_checkpoint_survives_saver_reconnection(tmp_path):
    database_path = tmp_path / "checkpoints.sqlite"
    thread_config = config("thread-persisted")

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver_a:
            graph_a = create_plan_execution_graph(
                create_default_tool_registry(),
                checkpointer=saver_a,
            )
            await graph_a.ainvoke(initial_state("C001"), config=thread_config)

        async with async_checkpoint_saver(database_path) as saver_b:
            graph_b = create_plan_execution_graph(
                create_default_tool_registry(),
                checkpointer=saver_b,
            )
            return await graph_b.aget_state(thread_config)

    snapshot = asyncio.run(scenario())

    assert snapshot.values["current_step_index"] == 1
    assert snapshot.values["route"] == "end"
    assert len(snapshot.values["step_results"]) == 1


def test_checkpoint_threads_are_isolated(tmp_path):
    database_path = tmp_path / "checkpoints.sqlite"

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver:
            graph = create_plan_execution_graph(
                create_default_tool_registry(),
                checkpointer=saver,
            )
            config_a = config("thread-a")
            config_b = config("thread-b")
            await graph.ainvoke(initial_state("C001"), config=config_a)
            await graph.ainvoke(initial_state("C002"), config=config_b)
            return (
                await graph.aget_state(config_a),
                await graph.aget_state(config_b),
            )

    snapshot_a, snapshot_b = asyncio.run(scenario())
    records_a = snapshot_a.values["step_results"][0]["result"]
    records_b = snapshot_b.values["step_results"][0]["result"]

    assert {record["id"] for record in records_a} == {"FB-001", "FB-003"}
    assert {record["id"] for record in records_b} == {"FB-005"}
    assert snapshot_a.values["goal"] != snapshot_b.values["goal"]


def test_checkpointed_graph_requires_thread_id(tmp_path):
    database_path = tmp_path / "checkpoints.sqlite"

    async def scenario():
        async with async_checkpoint_saver(database_path) as saver:
            graph = create_plan_execution_graph(
                create_default_tool_registry(),
                checkpointer=saver,
            )
            await graph.ainvoke(initial_state("C001"))

    with pytest.raises(ValueError):
        asyncio.run(scenario())
