from __future__ import annotations
import subprocess, sys
from pathlib import Path
from typing import Any, Dict

def sandbox_smoke(state: Dict[str, Any]) -> Dict[str, Any]:
    p = Path(state.get("scaffold", {}).get("path",""))
    logp = p / "reports" / "run1.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    try:
        code = "from src.agent.graph import graph; print('OK' if graph else 'NO')"
        res = subprocess.run([sys.executable, "-c", code], cwd=p, capture_output=True, text=True)
        logp.write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
        ok = ("OK" in (res.stdout or ""))
        if not ok:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=p, check=False)
            res2 = subprocess.run([sys.executable, "-c", code], cwd=p, capture_output=True, text=True)
            logp.write_text(res.stdout + "\n---RETRY---\n" + res2.stdout + "\n" + res2.stderr, encoding="utf-8")
            ok = ("OK" in (res2.stdout or ""))
    except Exception as e:
        logp.write_text(str(e), encoding="utf-8")
    state["sandbox"] = {"ok": ok, "log_path": str(logp)}
    return state
