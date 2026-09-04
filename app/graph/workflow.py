from langgraph.graph import END, START, StateGraph

from app.graph.nodes import create_llm_node
from app.graph.state import AgentState
from app.services.llm_service import LLMService


def create_basic_agent_graph(llm_service: LLMService):
    builder = StateGraph(AgentState)
    builder.add_node("llm", create_llm_node(llm_service))
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder.compile()
