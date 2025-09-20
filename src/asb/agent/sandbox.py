from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


@dataclass
class CommandResult:
    """Information captured for a single command execution."""

    argv: Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
        }


@dataclass
class Phase:
    """Represents a single sandbox validation phase."""

    name: str
    commands: Sequence[Sequence[str]]
    abort_on_fail: bool = True
    environment: Mapping[str, str] | None = None


@dataclass
class PhaseResult:
    """Aggregated results for a phase and the commands it executed."""

    name: str
    ok: bool
    commands: List[CommandResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "commands": [cmd.to_dict() for cmd in self.commands],
        }


def _default_phases() -> List[Phase]:
    """Return the default comprehensive sandbox phases."""

    python = sys.executable
    return [
        Phase(
            name="Dependency Installation",
            commands=[
                [python, "-m", "pip", "install", "-e", ".", "langgraph-cli[inmem]"]
            ],
        ),
        Phase(
            name="Packaging Sanity",
            commands=[[python, "-m", "pip", "check"]],
            abort_on_fail=False,
        ),
        Phase(
            name="Bytecode Compilation",
            commands=[[python, "-m", "compileall", "src"]],
            abort_on_fail=False,
        ),
        Phase(
            name="Test Suite",
            commands=[[python, "-m", "pytest"]],
        ),
    ]


def _run_command(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None,
    log: Any,
) -> CommandResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=dict(env) if env else None,
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - start
    except Exception as exc:  # pragma: no cover - defensive logging
        duration = time.perf_counter() - start
        log.write(f"    ! Command execution raised {exc!r} after {duration:.2f}s\n")
        return CommandResult(
            argv=list(argv),
            returncode=-1,
            stdout="",
            stderr=str(exc),
            duration=duration,
        )

    log.write(f"    $ {' '.join(argv)}\n")
    log.write(f"      return code: {completed.returncode}\n")
    log.write("      stdout:\n")
    if completed.stdout:
        log.write(_indent_text(completed.stdout, "        "))
    else:
        log.write("        <empty>\n")
    log.write("      stderr:\n")
    if completed.stderr:
        log.write(_indent_text(completed.stderr, "        "))
    else:
        log.write("        <empty>\n")
    log.write(f"      duration: {duration:.2f}s\n")

    return CommandResult(
        argv=list(argv),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration=duration,
    )


def _indent_text(text: str, prefix: str) -> str:
    stripped = text.rstrip("\n")
    if not stripped:
        return f"{prefix}<empty>\n"
    return "\n".join(f"{prefix}{line}" for line in stripped.splitlines()) + "\n"


def _cleanup_artifacts(project_root: Path, log: Any) -> None:
    """Remove common build/test artefacts to keep the sandbox clean."""

    removal_targets: Iterable[Path] = list(
        {
            project_root / "build",
            project_root / "dist",
            project_root / "reports" / "tmp",
            project_root / ".pytest_cache",
        }
    )
    removal_targets = list(removal_targets)
    removal_targets.extend(project_root.glob("*.egg-info"))

    for candidate in removal_targets:
        if not candidate.exists():
            continue
        try:
            if candidate.is_dir():
                shutil.rmtree(candidate)
                log.write(f"  cleaned directory: {candidate.relative_to(project_root)}\n")
            else:
                candidate.unlink()
                log.write(f"  removed file: {candidate.relative_to(project_root)}\n")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            log.write(
                f"  ! failed to remove {candidate}: {exc!r}\n"
            )

    # Remove stray __pycache__ directories
    for pycache in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            log.write(
                f"  cleaned directory: {pycache.relative_to(project_root)}\n"
            )
        except Exception as exc:  # pragma: no cover - best effort cleanup
            log.write(f"  ! failed to remove {pycache}: {exc!r}\n")


def comprehensive_sandbox_test(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Execute the comprehensive sandbox validation workflow.

    Parameters
    ----------
    state:
        The agent state. The project root is read from ``state["scaffold"]["path"]``.

    Returns
    -------
    MutableMapping[str, Any]
        The updated state containing sandbox execution results.
    """

    scaffold_info = state.get("scaffold", {}) if isinstance(state, Mapping) else {}
    project_root = Path(scaffold_info.get("path", "")).expanduser().resolve()

    log_path = project_root / "reports" / "comprehensive_test.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    phases = _default_phases()
    phase_results: List[PhaseResult] = []
    overall_ok = True

    with log_path.open("w", encoding="utf-8") as log:
        log.write("=== Comprehensive Sandbox Test ===\n")
        log.write(f"Started: {datetime.utcnow().isoformat()}Z\n")
        log.write(f"Project root: {project_root}\n\n")

        for phase in phases:
            log.write(f"## Phase: {phase.name}\n")
            commands_results: List[CommandResult] = []
            phase_ok = True

            for argv in phase.commands:
                cmd_result = _run_command(argv, project_root, phase.environment, log)
                commands_results.append(cmd_result)
                if cmd_result.returncode != 0:
                    phase_ok = False
                    overall_ok = False
                    log.write(
                        f"    ! Command failed with exit code {cmd_result.returncode}\n"
                    )
                    if phase.abort_on_fail:
                        log.write("    ! Aborting remaining phases due to failure.\n\n")
                        break
            else:
                log.write("    phase completed successfully.\n\n")

            phase_results.append(PhaseResult(phase.name, phase_ok, commands_results))
            if not phase_ok and phase.abort_on_fail:
                break

        log.write("\n--- Cleanup ---\n")
        _cleanup_artifacts(project_root, log)

        log.write("\n=== Summary ===\n")
        summary = {
            "phases": [pr.to_dict() for pr in phase_results],
            "overall_ok": overall_ok,
        }
        log.write(json.dumps(summary, indent=2))
        log.write("\n")
        log.write(f"Finished: {datetime.utcnow().isoformat()}Z\n")

    state["sandbox"] = {
        "ok": overall_ok,
        "log_path": str(log_path),
        "phases": [pr.to_dict() for pr in phase_results],
    }
    return state


sandbox_smoke = comprehensive_sandbox_test

__all__ = ["comprehensive_sandbox_test", "sandbox_smoke"]
