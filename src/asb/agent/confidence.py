from __future__ import annotations
from typing import Any, Dict
from asb_config.settings import get_settings

from asb.utils.message_utils import extract_last_message_content

def _structural_score(plan: Dict[str, Any]) -> float:
    nodes = plan.get("nodes", []) or []
    edges = plan.get("edges", []) or []
    steps = len(nodes)
    out_counts: Dict[str, int] = {}
    for e in edges:
        src = e.get("from") or e.get("from_")
        if not src: continue
        out_counts[src] = out_counts.get(src, 0) + 1
    branches = sum(1 for _, c in out_counts.items() if c > 1)
    score = 0.9
    score -= max(0, steps - 4) * 0.05
    score -= branches * 0.10
    return float(max(0.0, min(1.0, score)))

def _tool_coverage_score(user_text: str, plan: Dict[str, Any]) -> float:
    text = (user_text or "").lower()
    implies = {
        "web": any(k in text for k in ["search","browse","web","google","crawl"]),
        "git": any(k in text for k in ["git","branch","commit","pr"]),
        "test": any(k in text for k in ["test","pytest","unit test","ci","coverage"]),
        "file": any(k in text for k in ["read file","write file","patch","edit file","apply patch"]),
    }
    present = {"web": False, "git": False, "test": False, "file": False}
    for n in plan.get("nodes", []) or []:
        tool = (n.get("tool") or "").lower()
        if any(k in tool for k in ["web","search","browser","http"]): present["web"] = True
        if "git" in tool: present["git"] = True
        if any(k in tool for k in ["pytest","test","runner","ci"]): present["test"] = True
        if any(k in tool for k in ["file","fs.","apply_patch","write","read"]): present["file"] = True
    needed = [k for k, v in implies.items() if v]
    if not needed: return 1.0
    covered = sum(1 for k in needed if present.get(k, False))
    return covered / max(1, len(needed))

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
    coverage = _tool_coverage_score(user_text, plan)
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
        "coverage": round(coverage, 3),
        "prior": round(prior, 3),
        "weights": {"self": round(w_self,3), "struct": round(w_struct,3), "cov": round(w_cov,3), "prior": round(w_prior,3)},
        "final": round(confidence, 3),
    }

    plan["confidence"] = confidence
    state["plan"] = plan
    state["debug"] = debug
    return state
