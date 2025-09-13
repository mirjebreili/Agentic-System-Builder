from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def report(state: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(state.get("scaffold", {}).get("path", ""))
    summ = {
        "goal": (state.get("plan") or {}).get("goal",""),
        "plan": state.get("plan"),
        "confidence": (state.get("plan") or {}).get("confidence", None),
        "confidence_breakdown": state.get("debug", {}).get("confidence_terms"),
        "tests": state.get("tests"),
        "executor_passed": state.get("passed", False),
        "sandbox": state.get("sandbox"),
        "notes": "MVP run."
    }
    path.joinpath("reports/summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    state["report"] = {"ok": True, "summary_path": str(path / "reports/summary.json")}
    return state
