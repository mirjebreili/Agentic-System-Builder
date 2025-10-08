from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Dict, Any, List, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
import operator

from src.tools.registry import build_registry, get_question
from src.agents.planner import plan_candidates
from src.agents.hitl import hitl_node

class GraphState(TypedDict):
    initial_request: str
    question: str
    registry: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    messages: Annotated[List[AnyMessage], operator.add]
    approved_plan: List[str]

def start_new_plan(state: GraphState) -> Dict[str, Any]:
    initial_request = state["messages"][-1].content
    question = get_question(initial_request)
    return {
        "initial_request": initial_request,
        "question": question,
        "approved_plan": [],
        "candidates": [],
    }

def build_registry_node(state: GraphState) -> Dict[str, Any]:
    registry = build_registry(state["initial_request"])
    return {"registry": registry}

def planner_node(state: GraphState) -> Dict[str, Any]:
    planner_output = plan_candidates(state["question"], state["registry"])
    return {"candidates": planner_output.get("candidates", [])}

def process_hitl_feedback(state: GraphState) -> Dict[str, Any]:
    last_message_content = state["messages"][-1].content.strip().upper()

    if last_message_content.startswith("APPROVE"):
        try:
            index = int(last_message_content.split(" ")[1])
            approved_plan = state["candidates"][index]["plan"]
            return {"approved_plan": approved_plan}
        except (ValueError, IndexError):
            pass

    return {}

def should_continue(state: GraphState) -> str:
    if state.get("approved_plan"):
        return "end"

    last_message_content = state["messages"][-1].content.strip().upper()
    if last_message_content.startswith("REVISE"):
        return "replan"

    return "hitl"

def route_first_message(state: GraphState) -> str:
    if len(state.get("messages", [])) <= 1:
        return "new_plan"
    else:
        return "process_feedback"

def get_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("start_new_plan", start_new_plan)
    workflow.add_node("build_registry", build_registry_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("process_hitl_feedback", process_hitl_feedback)

    workflow.set_conditional_entry_point(
        route_first_message,
        {"new_plan": "start_new_plan", "process_feedback": "process_hitl_feedback"}
    )
    workflow.add_edge("start_new_plan", "build_registry")
    workflow.add_edge("build_registry", "planner")
    workflow.add_edge("planner", "hitl")
    workflow.add_edge("hitl", "process_hitl_feedback")

    workflow.add_conditional_edges(
        "process_hitl_feedback",
        should_continue,
        {"replan": "planner", "hitl": "hitl", "end": END}
    )

    # Add a memory saver to enable stateful interactions
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer, interrupt_after=["hitl"])