from typing import Dict, Any

def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    A node that represents the Human-in-the-Loop (HITL) step.

    This node prints the plan candidates to the console and then
    acts as a terminal point for the graph execution. The graph
    will pause here, returning its current state.
    """
    # Correctly access the candidates from the planner_output key in the state
    planner_output = state.get("planner_output", {})
    candidates = planner_output.get("candidates", [])

    print("\n--- HUMAN-IN-THE-LOOP (HITL) ---")
    print("The following plan candidates have been generated:")

    if not candidates:
        print("No plans to display.")
    else:
        for i, cand in enumerate(candidates):
            confidence = cand.get('confidence', 0)
            # Format confidence as percentage
            confidence_str = f"{confidence:.2%}" if isinstance(confidence, float) else "N/A"

            print(f"\nCandidate #{i} (Confidence: {confidence_str})")
            print(f"  - Plan: {cand.get('plan')}")
            print(f"  - Rationale: {cand.get('rationale')}")
            print(f"  - Scores: {cand.get('scores')}")

    print("\nGraph execution is paused for user review.")
    print("To proceed, the user would typically respond with 'APPROVE <index>' or 'REVISE <notes>'.")
    print("------------------------------------")

    # Pass the state through, making it the final output of the graph.
    return state