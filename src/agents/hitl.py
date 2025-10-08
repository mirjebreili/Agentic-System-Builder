from typing import Dict, Any, List

from src.utils.types import PlanCandidate


def format_candidates(candidates: List[PlanCandidate]) -> str:
    """Formats the plan candidates for display to the user."""
    if not candidates:
        return "No plan candidates were generated."

    output = ["Here are the proposed plans:"]
    for i, cand in enumerate(candidates):
        confidence = cand.get("confidence", 0)
        confidence_str = f"{confidence:.1%}"
        plan_str = " -> ".join(cand.get("plan", []))
        rationale = cand.get("rationale", "No rationale provided.")
        scores = cand.get("scores", {})
        scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())

        output.append(
            f"\nCandidate #{i} (Confidence: {confidence_str}):\n"
            f"  - Plan: {plan_str}\n"
            f"  - Rationale: {rationale}\n"
            f"  - Scores: [{scores_str}]"
        )
    return "\n".join(output)


def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Human-in-the-Loop (HITL) node.

    This node presents the plan candidates to the user and, in a fully
    interactive setup, would pause for user input. For this plan-only
    graph, it prints the plans and instructions, then terminates the graph.
    """
    planner_output = state.get("planner_output", {})
    candidates = planner_output.get("candidates", [])

    # Format and print the candidates for the user to review.
    display_message = format_candidates(candidates)
    print("\n--- HUMAN-IN-THE-LOOP (HITL) ---\n")
    print(display_message)
    print("\n------------------------------------")
    print("Graph execution is paused for user review.")
    print("To proceed, please respond with one of the following commands:")
    print("  - APPROVE <index>")
    print("  - REVISE <instructions>")
    print("------------------------------------\n")

    # In this version, the graph will end here.
    # The 'chosen' field from the planner already indicates the top-ranked plan.
    # A subsequent execution cycle would handle the APPROVE/REVISE logic.
    return {}