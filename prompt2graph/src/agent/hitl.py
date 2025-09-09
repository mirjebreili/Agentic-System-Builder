import logging
from typing import Optional, Dict, Any, Literal

from pydantic import BaseModel, ValidationError

from prompt2graph.config.settings import settings
from .state import AppState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Schema for HITL Resume Payload ---

class ReviewPlanPayload(BaseModel):
    """
    Defines the shape of the data the user sends back to the graph
    after a plan review interrupt.
    """
    action: Literal["approve", "revise"]
    plan: Optional[Dict[str, Any]] = None   # Include if approving with edits
    feedback: Optional[str] = None          # Include if revising


# --- HITL Node ---

def review_plan(state: AppState) -> AppState:
    """
    This node is a placeholder for the Human-in-the-Loop (HITL) review.

    In the graph definition, an interrupt is configured to trigger after this
    node executes. The graph will pause, allowing a human to review the plan.

    The user's response (the 'resume payload') is merged back into the state.
    The actual logic for handling the 'approve' or 'revise' action is
    implemented in the conditional edge that follows this node in the graph.
    """
    if not settings.require_plan_approval:
        logging.info("Skipping HITL plan review as per settings.")
        # If skipping, we inject a default "approve" action into the state
        # so the downstream routing logic can proceed.
        return {
            **state,
            "review": {"action": "approve", "feedback": "skipped"},
        }

    logging.info("Entering HITL plan review step. The graph will now pause for user input.")
    # No state change is needed here. The interrupt configured on this node
    # will handle the pause. The resume payload will be handled by the
    # subsequent conditional edge.
    return state


def process_review(state: AppState) -> AppState:
    """
    This function processes the review payload after the interrupt has been resumed.
    It's designed to be called by the conditional routing logic in the main graph.
    """
    review_payload = state.get("review")
    if not review_payload:
        # This case should ideally not be hit if routing is correct
        logging.warning("No review payload found. Defaulting to revision.")
        return {**state, "replan": True}

    try:
        review = ReviewPlanPayload.model_validate(review_payload)
        if review.action == "approve":
            logging.info("Plan approved by user.")
            # If the user edited the plan, update it in the state.
            updated_plan = review.plan or state.get("plan")
            return {
                **state,
                "plan": updated_plan,
                "replan": False,
            }
        else:  # revise
            logging.info(f"Plan revision requested. Feedback: {review.feedback}")
            # Add feedback to messages to inform the planner
            feedback_message = ChatMessage(role="system", content=f"User revision feedback: {review.feedback}")
            new_messages = state.get("messages", []) + [feedback_message]
            return {
                **state,
                "messages": new_messages,
                "replan": True,
            }
    except ValidationError as e:
        logging.error(f"Invalid resume payload: {e}. Defaulting to revision.")
        return {
            **state,
            "replan": True,
        }
