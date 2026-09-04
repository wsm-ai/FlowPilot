from typing import cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.execution_nodes import create_plan_step_executor
from app.graph.planning_nodes import create_approval_marker_node
from app.graph.routing import PlanRoute, route_plan_step
from app.graph.state import AgentState
from app.tools.registry import ToolRegistry


def _record_route(state: AgentState) -> dict[str, object]:
    return {"route": route_plan_step(state)}


def _route_from_state(state: AgentState) -> PlanRoute:
    return cast(PlanRoute, state["route"])


def create_plan_execution_graph(
    registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver | None = None,
):
    builder = StateGraph(AgentState)
    builder.add_node("router_entry", _record_route)
    builder.add_node("execute_step", create_plan_step_executor(registry))
    builder.add_node("approval_marker", create_approval_marker_node())
    builder.add_edge(START, "router_entry")
    builder.add_conditional_edges(
        "router_entry",
        _route_from_state,
        {
            "execute": "execute_step",
            "approval": "approval_marker",
            "end": END,
        },
    )
    builder.add_edge("execute_step", "router_entry")
    builder.add_edge("approval_marker", END)
    return builder.compile(checkpointer=checkpointer)
