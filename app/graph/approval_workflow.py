from typing import cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.approval_nodes import (
    create_approval_node,
    create_prepare_approval_node,
    create_reject_approval_node,
)
from app.graph.execution_nodes import (
    create_approved_plan_step_executor,
    create_plan_step_executor,
)
from app.graph.routing import (
    ApprovalDecisionRoute,
    PlanRoute,
    route_approval_decision,
    route_plan_step,
)
from app.graph.state import AgentState
from app.tools.registry import ToolRegistry


def _record_route(state: AgentState) -> dict[str, object]:
    return {"route": route_plan_step(state)}


def _route_from_state(state: AgentState) -> PlanRoute:
    return cast(PlanRoute, state["route"])


def _route_from_approval_decision(state: AgentState) -> ApprovalDecisionRoute:
    return route_approval_decision(state)


def create_approval_graph(
    registry: ToolRegistry,
    checkpointer: BaseCheckpointSaver,
):
    builder = StateGraph(AgentState)
    builder.add_node("router_entry", _record_route)
    builder.add_node("execute_step", create_plan_step_executor(registry))
    builder.add_node("prepare_approval", create_prepare_approval_node())
    builder.add_node("approval_node", create_approval_node())
    builder.add_node(
        "approved_execute",
        create_approved_plan_step_executor(registry),
    )
    builder.add_node("reject_node", create_reject_approval_node())
    builder.add_edge(START, "router_entry")
    builder.add_conditional_edges(
        "router_entry",
        _route_from_state,
        {
            "execute": "execute_step",
            "approval": "prepare_approval",
            "end": END,
        },
    )
    builder.add_edge("execute_step", "router_entry")
    builder.add_edge("prepare_approval", "approval_node")
    builder.add_conditional_edges(
        "approval_node",
        _route_from_approval_decision,
        {
            "approve": "approved_execute",
            "reject": "reject_node",
        },
    )
    builder.add_edge("approved_execute", "router_entry")
    builder.add_edge("reject_node", END)
    return builder.compile(checkpointer=checkpointer)
