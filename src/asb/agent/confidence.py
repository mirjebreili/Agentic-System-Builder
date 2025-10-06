from __future__ import annotations
import re
from typing import Any, Dict
from asb_config.settings import get_settings

from asb.utils.message_utils import extract_last_message_content

def _structural_score(plan: Dict[str, Any]) -> float:
    capabilities = plan.get("capabilities", []) or []
    if not capabilities:
        return 0.1

    count = len(capabilities)
    score = 0.95
    if count < 2:
        score -= 0.2
    if count > 6:
        score -= min(0.3, (count - 6) * 0.05)

    missing_desc = sum(1 for c in capabilities if not (c.get("description") or "").strip())
    weak_terms = sum(1 for c in capabilities if len(c.get("search_terms") or []) < 2)
    missing_ecosystems = sum(1 for c in capabilities if not (c.get("ecosystem_priority") or []))

    score -= missing_desc * 0.05
    score -= weak_terms * 0.07
    score -= missing_ecosystems * 0.05

    return float(max(0.0, min(1.0, score)))

def _alignment_score(user_text: str, plan: Dict[str, Any]) -> float:
    capabilities = plan.get("capabilities", []) or []
    if not capabilities:
        return 0.2

    keywords = {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", (user_text or "").lower())
        if len(token) >= 4
    }
    if not keywords:
        return 1.0

    matches = 0
    for capability in capabilities:
        haystack = " ".join(
            [
                capability.get("name", ""),
                capability.get("description", ""),
                " ".join(capability.get("search_terms", []) or []),
            ]
        ).lower()
        if any(keyword in haystack for keyword in keywords):
            matches += 1

    return matches / max(1, len(capabilities))

def _prior_success_score(metrics: Dict[str, Any]) -> float:
    succ = int(metrics.get("prior_successes", 0))
    att = int(metrics.get("prior_attempts", 0))
    return (succ + 1) / (att + 2)

def compute_plan_confidence(state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_settings()
    plan = dict(state.get("plan") or {})
    messages = state.get("messages") or []
    user_text = extract_last_message_content(messages, "") if messages else ""
    metrics = dict(state.get("metrics") or {})

    self_score = float(plan.get("confidence", 0.0) or 0.0)
    structural = _structural_score(plan)
    coverage = _alignment_score(user_text, plan)
    prior = _prior_success_score(metrics)

    w_self = float(getattr(cfg, "conf_w_self", 0.4))
    w_struct = float(getattr(cfg, "conf_w_struct", 0.25))
    w_cov = float(getattr(cfg, "conf_w_cov", 0.2))
    w_prior = float(getattr(cfg, "conf_w_prior", 0.15))
    w_sum = max(1e-9, w_self + w_struct + w_cov + w_prior)
    w_self, w_struct, w_cov, w_prior = (w_self / w_sum, w_struct / w_sum, w_cov / w_sum, w_prior / w_sum)

    confidence = w_self*self_score + w_struct*structural + w_cov*coverage + w_prior*prior
    confidence = float(max(0.0, min(1.0, confidence)))

    debug = dict(state.get("debug") or {})
    debug["confidence_terms"] = {
        "self": round(self_score, 3),
        "structural": round(structural, 3),
        "alignment": round(coverage, 3),
        "prior": round(prior, 3),
        "weights": {"self": round(w_self,3), "struct": round(w_struct,3), "align": round(w_cov,3), "prior": round(w_prior,3)},
        "final": round(confidence, 3),
    }

    plan["confidence"] = confidence
    state["plan"] = plan
    state["debug"] = debug
    return state
