from typing import Dict, Any, List, Optional
import os

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


def _parse_user_reply(reply: str) -> Dict[str, Optional[str]]:
    """Parses a user's reply string into action and payload.

    Returns a dict with keys: action (APPROVE|REVISE|NONE) and payload (index or instructions).
    """
    if not reply:
        return {"action": None, "payload": None}

    reply = reply.strip()
    if reply.upper().startswith("APPROVE"):
        parts = reply.split(maxsplit=1)
        if len(parts) == 2:
            return {"action": "APPROVE", "payload": parts[1].strip()}
        return {"action": "APPROVE", "payload": None}
    if reply.upper().startswith("REVISE"):
        parts = reply.split(maxsplit=1)
        if len(parts) == 2:
            return {"action": "REVISE", "payload": parts[1].strip()}
        return {"action": "REVISE", "payload": None}

    return {"action": None, "payload": reply}


def hitl_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Human-in-the-Loop (HITL) node.

    Behavior:
    - Presents the plan candidates.
    - If `user_reply` is present in the state, parses it and acts:
      - APPROVE <index>: sets `current_plan` to the approved candidate's plan.
      - REVISE <instructions>: returns a flag `require_replan` with instructions.
    - If no `user_reply`, prints the candidates and instructions and returns without changing `current_plan`.
    """
    planner_output = state.get("planner_output", {})
    candidates: List[PlanCandidate] = planner_output.get("candidates", [])

    # Format and print the candidates for the user to review.
    display_message = format_candidates(candidates)
    print("\n--- HUMAN-IN-THE-LOOP (HITL) ---\n")
    print(display_message)
    print("\n------------------------------------")
    print("To proceed, set `user_reply` in the state and re-invoke the graph with one of:")
    print("  - APPROVE <index>")
    print("  - REVISE <instructions>")
    print("------------------------------------\n")

    # Check for a user reply in the state.
    user_reply = state.get("user_reply")

    # If interactive mode is enabled (via state or env), prompt for input when no user_reply
    interactive_flag = state.get("interactive") or os.getenv("HITL_INTERACTIVE") == "1"
    if not user_reply and interactive_flag:
        try:
            user_reply = input("HITL> ").strip()
        except Exception:
            user_reply = None

    if not user_reply:
        # No action - preserve existing current_plan
        return {}

    parsed = _parse_user_reply(user_reply)
    action = parsed.get("action")
    payload = parsed.get("payload")

    result: Dict[str, Any] = {}

    if action == "APPROVE":
        # Determine index; if none provided, use planner's chosen index if available.
        try:
            if payload is None:
                idx = int(planner_output.get("chosen", 0))
            else:
                idx = int(payload)
        except Exception:
            print("Invalid APPROVE index; no change made.")
            return {}

        if 0 <= idx < len(candidates):
            approved = candidates[idx]
            # Store as a list to match GraphState typing (list of PlanCandidate)
            result["current_plan"] = [approved]
            print(f"Plan #{idx} approved.")
        else:
            print("APPROVE index out of range; no change made.")

    elif action == "REVISE":
        # Inform the graph that a replan is requested with instructions.
        result["require_replan"] = payload or ""
        print("Revision requested:", payload)

    else:
        # Unrecognized action; do nothing.
        print("Unrecognized user action; no change made.")

    return result