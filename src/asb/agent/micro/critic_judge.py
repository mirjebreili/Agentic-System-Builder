"""Pick the most promising Tree-of-Thought variant."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _score_variant(variant: Dict[str, Any]) -> Tuple[float, str]:
    code = variant.get("code") or ""
    path = variant.get("path") or "unknown"
    confidence = float(variant.get("confidence", 0.0))
    score = confidence
    reason = [f"base={confidence:.2f}"]
    try:
        compile(code, path, "exec")
    except SyntaxError as exc:
        reason.append(f"syntax_error:{exc.lineno}")
        score -= 0.5
    else:
        score += 0.2
        reason.append("syntax_ok")
    if "AIMessage" in code:
        score += 0.05
        reason.append("messages")
    if "return updated_state" in code:
        score += 0.05
        reason.append("returns_state")
    return score, ",".join(reason)


def critic_judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select the highest scoring variant for downstream patching."""

    working_state: Dict[str, Any] = dict(state or {})
    scratch = dict(working_state.get("scratch") or {})
    variants: List[Dict[str, Any]] = list(scratch.get("tot_variants") or [])
    if not variants:
        scratch["selected_variants"] = []
        working_state["scratch"] = scratch
        return working_state

    evaluated: List[Tuple[float, Dict[str, Any], str]] = []
    for variant in variants:
        score, details = _score_variant(variant)
        evaluated.append((score, variant, details))
    evaluated.sort(key=lambda item: item[0], reverse=True)

    best_score, best_variant, rationale = evaluated[0]
    selected = dict(best_variant)
    selected["score"] = best_score
    selected["rationale"] = rationale
    scratch["selected_variants"] = [selected]
    scratch.setdefault("tot_debug", {})["evaluated"] = [
        {"path": variant.get("path"), "score": score, "details": reason}
        for score, variant, reason in evaluated
    ]
    working_state["scratch"] = scratch
    return working_state


__all__ = ["critic_judge_node"]
