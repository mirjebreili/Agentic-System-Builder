from __future__ import annotations

"""Validation utilities for generated Agentic System Builder projects."""

from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class ValidationResult:
    """Container for a validation check outcome."""

    name: str
    success: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "success": self.success, **self.details}


class CodeValidator:
    """Deep validation agent for generated code projects.

    The validator performs a series of structural and runtime checks intended to
    surface the most common issues observed in generated projects.  Each check is
    deliberately isolated to provide actionable feedback that can be consumed by a
    follow-up fixer agent.
    """

    def validate_project(self, project_path: Path) -> Dict[str, Any]:
        """Run the full validation suite for ``project_path``."""

        results: List[ValidationResult] = []
        results.append(ValidationResult("structure_check", *self._check_project_structure(project_path)))
        results.append(ValidationResult("dependency_check", *self._check_dependencies(project_path)))
        results.append(ValidationResult("import_check", *self._check_imports(project_path)))
        results.append(ValidationResult("langgraph_check", *self._check_langgraph_compatibility(project_path)))
        results.append(ValidationResult("runtime_check", *self._check_runtime_execution(project_path)))

        summary: Dict[str, Any] = {
            "checks": [result.to_dict() for result in results],
        }

        failures = [result for result in results if not result.success]
        summary["overall_success"] = not failures
        summary["success"] = summary["overall_success"]
        summary["errors"] = [result.details.get("message") for result in failures if result.details.get("message")]
        summary["fixes_needed"] = [result.name for result in failures]
        return summary

    # ------------------------------------------------------------------
    # Individual validation phases
    # ------------------------------------------------------------------
    def _check_project_structure(self, project_path: Path) -> Tuple[bool, Dict[str, Any]]:
        required_files = [
            "langgraph.json",
            "pyproject.toml",
            "README.md",
            "src/agent/graph.py",
            "src/agent/planner.py",
            "src/agent/executor.py",
        ]

        missing = [file_path for file_path in required_files if not (project_path / file_path).exists()]
        success = not missing
        return success, {
            "missing_files": missing,
            "message": "Structure is valid" if success else f"Missing {len(missing)} required files",
        }

    def _check_dependencies(self, project_path: Path) -> Tuple[bool, Dict[str, Any]]:
        try:
            python = sys.executable
            with tempfile.TemporaryDirectory() as temp_dir:
                venv_path = Path(temp_dir) / "validation_venv"
                subprocess.run([python, "-m", "venv", str(venv_path)], check=True, capture_output=True)

                pip_path = venv_path / ("Scripts" if os.name == "nt" else "bin") / "pip"
                completed = subprocess.run(
                    [str(pip_path), "install", "-e", ".", "--quiet"],
                    cwd=str(project_path),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                success = completed.returncode == 0
                message = "Dependencies installed successfully" if success else "Dependency installation failed"
                return success, {
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "message": message,
                }
        except Exception as exc:  # pragma: no cover - defensive branch
            return False, {"error": str(exc), "message": f"Dependency check failed: {exc}"}

    def _check_imports(self, project_path: Path) -> Tuple[bool, Dict[str, Any]]:
        project_src = project_path / "src"
        failures: List[Tuple[str, str]] = []
        original_sys_path = list(sys.path)
        try:
            sys.path.insert(0, str(project_src))
            test_imports = [
                "agent.graph",
                "agent.planner",
                "agent.executor",
                "config.settings",
                "llm.client",
            ]

            for module in test_imports:
                try:
                    __import__(module)
                except Exception as exc:  # broad to capture runtime errors
                    failures.append((module, repr(exc)))

            success = not failures
            message = "All imports resolved" if success else f"Import check failed for {len(failures)} modules"
            return success, {"failed_imports": failures, "message": message}
        finally:
            sys.path[:] = original_sys_path

    def _check_langgraph_compatibility(self, project_path: Path) -> Tuple[bool, Dict[str, Any]]:
        langgraph_cli = shutil.which("langgraph")
        if not langgraph_cli:
            return True, {
                "message": "LangGraph CLI not found in PATH",
                "stdout": "",
                "stderr": "",
                "skipped": True,
            }

        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                [langgraph_cli, "dev", "--port", "9999", "--no-browser"],
                cwd=str(project_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(5)
            running = process.poll() is None
            if running:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

            stdout, stderr = process.communicate(timeout=5)
            success = running and not stderr.decode().strip()
            message = "LangGraph dev started successfully" if success else "LangGraph dev failed to start"
            return success, {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "message": message,
                "skipped": False,
            }
        except FileNotFoundError:
            return True, {
                "message": "LangGraph CLI not available",
                "stdout": "",
                "stderr": "",
                "skipped": True,
            }
        except Exception as exc:  # pragma: no cover - defensive logging
            return False, {"message": f"LangGraph compatibility test failed: {exc}", "error": str(exc)}
        finally:
            if process and process.poll() is None:
                process.kill()

    def _check_runtime_execution(self, project_path: Path) -> Tuple[bool, Dict[str, Any]]:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "compileall", "src"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                timeout=60,
            )
            success = completed.returncode == 0
            message = "Runtime compilation succeeded" if success else "Runtime compilation failed"
            return success, {"stdout": completed.stdout, "stderr": completed.stderr, "message": message}
        except Exception as exc:  # pragma: no cover - defensive branch
            return False, {"message": f"Runtime check failed: {exc}", "error": str(exc)}


def code_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Code validator node for the ASB execution graph."""

    scaffold_info = state.get("scaffold", {}) if isinstance(state, dict) else {}
    project_path = Path(scaffold_info.get("path", ""))
    if not project_path.exists():
        state["code_validation"] = {
            "success": False,
            "error": "Project path not found",
            "next_action": "regenerate",
        }
        state["next_action"] = "fix_code"
        state["validation_errors"] = ["project path not found"]
        return state

    validator = CodeValidator()
    results = validator.validate_project(project_path)
    state["code_validation"] = results
    state["validation_errors"] = results.get("errors", [])
    state["next_action"] = "complete" if results.get("overall_success") else "fix_code"
    return state
