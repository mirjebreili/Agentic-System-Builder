from __future__ import annotations
import subprocess, sys
from pathlib import Path
from typing import Any, Dict

def sandbox_smoke(state: Dict[str, Any]) -> Dict[str, Any]:
    p = Path(state.get("scaffold", {}).get("path", ""))
    logp = p / "reports" / "run1.log"
    logp.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    try:
        # Install dependencies first
        install_cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "langgraph-cli[inmem]"]
        install_res = subprocess.run(install_cmd, cwd=p, capture_output=True, text=True, check=False)

        # Write installation logs
        log_content = f"--- Installation ---\n"
        log_content += f"STDOUT:\n{install_res.stdout}\n"
        log_content += f"STDERR:\n{install_res.stderr}\n"

        if install_res.returncode == 0:
            # Run tests using pytest
            test_cmd = [sys.executable, "-m", "pytest"]
            test_res = subprocess.run(test_cmd, cwd=p, capture_output=True, text=True)

            log_content += f"\n--- Pytest ---\n"
            log_content += f"STDOUT:\n{test_res.stdout}\n"
            log_content += f"STDERR:\n{test_res.stderr}\n"

            ok = test_res.returncode == 0
        else:
            log_content += "\n--- Installation failed ---"

        logp.write_text(log_content, encoding="utf-8")

    except Exception as e:
        logp.write_text(f"An unexpected error occurred: {e}", encoding="utf-8")

    state["sandbox"] = {"ok": ok, "log_path": str(logp)}
    return state
