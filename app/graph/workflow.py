from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    create_final_llm_node,
    create_llm_node,
    create_tool_node,
    route_after_llm,
)
from app.graph.state import AgentState
from app.services.llm_service import LLMService
from app.tools.registry import ToolRegistry


def create_basic_agent_graph(llm_service: LLMService, registry: ToolRegistry):
    tools = registry.definitions()
    builder = StateGraph(AgentState)
    builder.add_node(
        "llm",
        create_llm_node(llm_service, tools),
    )
    builder.add_node("tools", create_tool_node(registry))
    builder.add_node("final_llm", create_final_llm_node(llm_service, tools))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "tools": "tools",
            "end": END,
        },
    )
    builder.add_edge("tools", "final_llm")
    builder.add_edge("final_llm", END)
    return builder.compile()
