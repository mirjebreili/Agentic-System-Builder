from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from langgraph.types import interrupt

from ..utils.types import PlanCandidate, PlannerState, PlanningResult

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "hitl.md"
_HITL_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8").strip()


def _format_candidate(index: int, candidate: PlanCandidate) -> str:
    plan_chain = " → ".join(candidate.plan)
    return (
        f"{index}. Plan: {plan_chain}\n"
        f"   Rationale: {candidate.rationale}\n"
        f"   Confidence: {candidate.confidence:.2f} | Raw score: {candidate.raw_score:.4f}"
    )


def _format_summary(result: PlanningResult) -> str:
    lines = ["Candidate plans:"]
    for idx, candidate in enumerate(result["candidates"]):
        lines.append(_format_candidate(idx, candidate))
    return "\n".join(lines)


def _parse_response(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        if "action" in value:
            action = str(value["action"]).strip().upper()
            payload = value.get("value") or value.get("plan") or value.get("instructions")
            return {"action": action, "payload": payload}
        if "command" in value:
            action = str(value["command"]).strip().upper()
            payload = value.get("payload")
            return {"action": action, "payload": payload}

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {"action": "", "payload": ""}
        upper = text.upper()
        if upper.startswith("APPROVE"):
            parts = text.split(maxsplit=1)
            payload = parts[1] if len(parts) > 1 else ""
            return {"action": "APPROVE", "payload": payload}
        if upper.startswith("REVISE"):
            parts = text.split(maxsplit=1)
            payload = parts[1] if len(parts) > 1 else ""
            return {"action": "REVISE", "payload": payload}
        return {"action": text.upper(), "payload": text}

    return {"action": "", "payload": value}


def perform_hitl(state: PlannerState) -> Dict[str, Any]:
    result = state.get("planner_result")
    if not result:
        return {"hitl_status": "skipped"}

    summary = _format_summary(result)
    prompt_text = f"{_HITL_PROMPT}\n\n{summary}".strip()

    payload: Dict[str, Any] = {"prompt": prompt_text, "summary": summary}

    while True:
        response = interrupt(payload)
        parsed = _parse_response(response)
        action = parsed.get("action", "").upper()

        if action == "APPROVE":
            raw_index = str(parsed.get("payload", "")).strip()
            try:
                index = int(raw_index)
            except ValueError:
                payload = {
                    "prompt": "Invalid APPROVE index. Reply with APPROVE <index> (e.g., APPROVE 0).",
                    "summary": summary,
                }
                continue
            if index < 0 or index >= len(result["candidates"]):
                payload = {
                    "prompt": f"Index {index} is out of range. Valid options: 0..{len(result['candidates']) - 1}.",
                    "summary": summary,
                }
                continue
            chosen_candidate = result["candidates"][index]
            return {
                "hitl_status": "approved",
                "approved_plan": list(chosen_candidate.plan),
                "pending_plan": list(chosen_candidate.plan),
                "hitl_prompt": prompt_text,
            }

        if action == "REVISE":
            feedback = parsed.get("payload", "")
            return {
                "hitl_status": "revise",
                "hitl_feedback": feedback,
                "hitl_prompt": prompt_text,
            }

        payload = {
            "prompt": "Unrecognized response. Please reply with APPROVE <index> or REVISE <notes>.",
            "summary": summary,
        }
