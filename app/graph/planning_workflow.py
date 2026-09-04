from langgraph.graph import END, START, StateGraph

from app.graph.planning_nodes import create_planner_node
from app.graph.state import AgentState
from app.services.planner_service import PlannerService


def create_planning_graph(planner_service: PlannerService):
    builder = StateGraph(AgentState)
    builder.add_node("planner", create_planner_node(planner_service))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)
    return builder.compile()
