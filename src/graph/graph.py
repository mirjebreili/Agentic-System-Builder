import json
from langgraph.graph import StatefulGraph, START, END
from typing import Dict, Any

from src.utils.types import GraphState
from src.llm.provider import call_llm
from src.tools.adapters.http_based_atlas_read_by_key import METADATA as http_tool_metadata
from src.tools.adapters.membased_atlas_key_stream_aggregator import METADATA as mem_tool_metadata

# Node definitions

def ingest_first_message(state: GraphState) -> Dict[str, Any]:
    """
    The input to the graph is the initial message. LangGraph will place it
    in the `initial_message` field of the state if the input key matches.
    This node is a placeholder to start the graph.
    """
    return {}


def build_registry(state: GraphState) -> Dict[str, Any]:
    """
    Builds the tool registry from the available tool metadata.
    """
    registry = {
        http_tool_metadata["name"]: http_tool_metadata,
        mem_tool_metadata["name"]: mem_tool_metadata,
    }
    return {"tool_registry": registry}


def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Calls the LLM to generate a plan.
    """
    # The stubbed LLM returns the candidates directly, so we don't need a real prompt.
    llm_response_str = call_llm(prompt="", vars={})
    llm_response = json.loads(llm_response_str)

    return {"candidates": llm_response.get("candidates", [])}


def hitl_node(state: GraphState) -> Dict[str, Any]:
    """
    Placeholder for Human-in-the-Loop review. Sets the final output.
    """
    return {
        "chosen": 0,
        "note": "stub pipeline — logic to be implemented in Task 2+"
    }


def get_graph():
    """
    Wires up the graph and compiles it.
    """
    workflow = StatefulGraph(GraphState)

    # Add nodes
    workflow.add_node("ingest_first_message", ingest_first_message)
    workflow.add_node("build_registry", build_registry)
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("hitl_node", hitl_node)

    # Define edges
    workflow.add_edge(START, "ingest_first_message")
    workflow.add_edge("ingest_first_message", "build_registry")
    workflow.add_edge("build_registry", "planner_node")
    workflow.add_edge("planner_node", "hitl_node")
    workflow.add_edge("hitl_node", END)

    # Compile the graph
    app = workflow.compile()

    # The client should send a JSON with an "initial_message" key.
    # e.g., curl -X POST -H "Content-Type: application/json" \
    # -d '{"input": {"initial_message": "hello"}}' \
    # http://127.0.0.1:8000/planner/invoke

    return app