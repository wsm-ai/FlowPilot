from typing import cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.approval_nodes import (
    create_approval_node,
    create_prepare_approval_node,
)
from app.graph.planning_nodes import create_execute_marker_node
from app.graph.routing import PlanRoute, route_plan_step
from app.graph.state import AgentState


def _record_route(state: AgentState) -> dict[str, object]:
    return {"route": route_plan_step(state)}


def _route_from_state(state: AgentState) -> PlanRoute:
    return cast(PlanRoute, state["route"])


def create_approval_graph(checkpointer: BaseCheckpointSaver):
    builder = StateGraph(AgentState)
    builder.add_node("router_entry", _record_route)
    builder.add_node("execute_marker", create_execute_marker_node())
    builder.add_node("prepare_approval", create_prepare_approval_node())
    builder.add_node("approval_node", create_approval_node())
    builder.add_edge(START, "router_entry")
    builder.add_conditional_edges(
        "router_entry",
        _route_from_state,
        {
            "execute": "execute_marker",
            "approval": "prepare_approval",
            "end": END,
        },
    )
    builder.add_edge("execute_marker", END)
    builder.add_edge("prepare_approval", "approval_node")
    builder.add_edge("approval_node", END)
    return builder.compile(checkpointer=checkpointer)
