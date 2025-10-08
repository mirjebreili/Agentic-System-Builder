from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, List, Optional

from src.tools.registry import build_registry, get_question
from src.agents.planner import plan_candidates
from src.agents.hitl import hitl_node
from src.utils.types import Registry, PlanCandidate


# Define the state for the graph
class GraphState(TypedDict):
    first_message: str
    question: str
    registry: Registry
    planner_output: Dict[str, Any]
    # Keep track of the current, approved plan.
    current_plan: Optional[List[PlanCandidate]]


def ingest_first_message(state: GraphState) -> Dict[str, Any]:
    """
    Node to ingest the first message and extract the question.
    This is the entry point of the graph.
    """
    first_message = state["first_message"]
    question = get_question(first_message)
    return {"question": question, "first_message": first_message}


def build_registry_node(state: GraphState) -> Dict[str, Any]:
    """
    Node to build the tool registry from the first message.
    """
    registry = build_registry(state["first_message"])
    return {"registry": registry}


def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Node to run the planner agent, generating candidate plans.
    """
    planner_output = plan_candidates(state["question"], state["registry"])
    return {"planner_output": planner_output}


def get_graph():
    """
    Builds and returns the LangGraph for the plan-only agent.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("ingest_first_message", ingest_first_message)
    workflow.add_node("build_registry", build_registry_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("hitl", hitl_node)

    # Define edges
    workflow.set_entry_point("ingest_first_message")
    workflow.add_edge("ingest_first_message", "build_registry")
    workflow.add_edge("build_registry", "planner")
    workflow.add_edge("planner", "hitl")

    # The HITL node is the end of this plan-only graph.
    # In a stateful version, there would be conditional edges based on user input.
    workflow.add_edge("hitl", END)

    # Compile the graph
    return workflow.compile()