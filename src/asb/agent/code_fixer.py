from __future__ import annotations

from __future__ import annotations

"""Automated remediation utilities for generated ASB projects."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class FixStrategy:
    """Representation of a potential remediation strategy."""

    name: str
    priority: str
    actions: List[str]
    feasibility_score: float = 0.0
    impact_score: float = 0.0

    def overall_score(self) -> float:
        return (self.feasibility_score + self.impact_score) / 2


class CodeFixer:
    """Tree-of-Thought inspired fixer for generated projects."""

    MAX_FIX_ATTEMPTS = 3

    def fix_project_issues(self, state: Dict[str, Any]) -> Dict[str, Any]:
        project_path = Path(state.get("scaffold", {}).get("path", ""))
        validation_results = state.get("code_validation", {})

        attempts = int(state.get("fix_attempts", 0)) + 1
        state["fix_attempts"] = attempts

        if attempts >= self.MAX_FIX_ATTEMPTS:
            print(
                f"🛑 CIRCUIT BREAKER: Max attempts ({attempts}) reached - FORCING COMPLETION"
            )
            fixes = {
                "success": False,
                "applied_fixes": [],
                "errors": [
                    "Max fix attempts exceeded - forcing completion",
                ],
            }
            selected = None
            state["next_action"] = "force_complete"
            state["code_fixes"] = fixes
            state["fix_strategy_used"] = None
            return state
        else:
            strategies = self._generate_fix_strategies(validation_results)
            evaluated = self._evaluate_strategies(strategies, project_path)
            selected = self._select_best_strategy(evaluated)
            fixes = self._apply_fixes(selected, project_path)

            state["next_action"] = (
                "validate_again" if fixes.get("success") else "manual_review"
            )

        state["code_fixes"] = fixes
        state["fix_strategy_used"] = selected.name if selected else None

        if state.get("next_action") != "validate_again":
            state.pop("fix_attempts", None)

        return state

    # ------------------------------------------------------------------
    # Strategy generation and evaluation
    # ------------------------------------------------------------------
    def _generate_fix_strategies(self, validation_results: Dict[str, Any]) -> List[FixStrategy]:
        strategies: List[FixStrategy] = []

        if not validation_results.get("import_check", {}).get("success", True):
            strategies.append(
                FixStrategy(
                    name="fix_imports",
                    priority="high",
                    actions=[
                        "update_langgraph_json_paths",
                        "fix_relative_imports",
                        "ensure_init_files",
                    ],
                )
            )

        if not validation_results.get("dependency_check", {}).get("success", True):
            strategies.append(
                FixStrategy(
                    name="fix_dependencies",
                    priority="high",
                    actions=[
                        "update_pyproject_toml",
                        "pin_dependency_versions",
                        "add_missing_packages",
                    ],
                )
            )

        if not validation_results.get("structure_check", {}).get("success", True):
            strategies.append(
                FixStrategy(
                    name="fix_structure",
                    priority="medium",
                    actions=[
                        "create_missing_files",
                        "fix_directory_structure",
                        "update_configuration",
                    ],
                )
            )

        if not strategies:
            strategies.append(FixStrategy(name="noop", priority="low", actions=[]))

        return strategies

    def _evaluate_strategies(self, strategies: List[FixStrategy], project_path: Path) -> List[FixStrategy]:
        for strategy in strategies:
            strategy.feasibility_score = self._calculate_feasibility(strategy, project_path)
            strategy.impact_score = self._calculate_impact(strategy)
        return sorted(strategies, key=lambda s: s.overall_score(), reverse=True)

    def _select_best_strategy(self, strategies: List[FixStrategy]) -> FixStrategy:
        return strategies[0]

    def _calculate_feasibility(self, strategy: FixStrategy, project_path: Path) -> float:
        if strategy.name == "noop":
            return 0.0
        if not project_path.exists():
            return 0.1
        existing_actions = sum(1 for action in strategy.actions if self._action_available(action))
        return existing_actions / max(len(strategy.actions), 1)

    def _calculate_impact(self, strategy: FixStrategy) -> float:
        priority_weights = {"high": 1.0, "medium": 0.7, "low": 0.3}
        return priority_weights.get(strategy.priority, 0.5)

    # ------------------------------------------------------------------
    # Fix application
    # ------------------------------------------------------------------
    def _apply_fixes(self, strategy: FixStrategy, project_path: Path) -> Dict[str, Any]:
        if strategy.name == "noop":
            return {"success": True, "applied_fixes": [], "errors": ["No fixes required"]}

        applied: List[str] = []
        errors: List[str] = []

        for action in strategy.actions:
            if not self._action_available(action):
                errors.append(f"No implementation for action '{action}'")
                continue
            try:
                getattr(self, f"_{action}")(project_path)
                applied.append(action)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to apply fix '%s'", action)
                errors.append(f"Fix {action} failed: {exc}")

        success = not errors
        return {"success": success, "applied_fixes": applied, "errors": errors}

    def _action_available(self, action: str) -> bool:
        return hasattr(self, f"_{action}")

    # ------------------------------------------------------------------
    # Individual fix implementations
    # ------------------------------------------------------------------
    def _update_langgraph_json_paths(self, project_path: Path) -> None:
        langgraph_file = project_path / "langgraph.json"
        if not langgraph_file.exists():
            raise FileNotFoundError("langgraph.json not found")

        data = json.loads(langgraph_file.read_text(encoding="utf-8"))
        graphs = data.setdefault("graphs", {})
        agent_path = graphs.get("agent")
        if isinstance(agent_path, str) and not agent_path.startswith("src."):
            graphs["agent"] = f"src.{agent_path}"
        langgraph_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _fix_relative_imports(self, project_path: Path) -> None:
        tests_dir = project_path / "tests"
        if not tests_dir.exists():
            return
        for test_file in tests_dir.glob("test_*.py"):
            content = test_file.read_text(encoding="utf-8")
            if "from agent.graph import graph" in content:
                content = content.replace("from agent.graph import graph", "from src.agent.graph import graph")
                test_file.write_text(content, encoding="utf-8")

    def _ensure_init_files(self, project_path: Path) -> None:
        packages = [
            project_path / "src",
            project_path / "src" / "agent",
            project_path / "src" / "config",
            project_path / "src" / "llm",
        ]
        for package_dir in packages:
            package_dir.mkdir(parents=True, exist_ok=True)
            init_file = package_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("", encoding="utf-8")

    # Placeholder implementations for dependency/structure fixes
    def _update_pyproject_toml(self, project_path: Path) -> None:  # pragma: no cover - heuristic action
        pyproject = project_path / "pyproject.toml"
        if not pyproject.exists():
            return
        text = pyproject.read_text(encoding="utf-8")
        if "requests" not in text:
            text = text.replace("dependencies = [", "dependencies = [\n  \"requests>=2.25.0\",")
        pyproject.write_text(text, encoding="utf-8")

    def _pin_dependency_versions(self, project_path: Path) -> None:  # pragma: no cover - heuristic action
        # Best-effort placeholder: nothing to do presently.
        return

    def _add_missing_packages(self, project_path: Path) -> None:  # pragma: no cover - heuristic action
        # Best-effort placeholder: nothing to do presently.
        return

    def _create_missing_files(self, project_path: Path) -> None:  # pragma: no cover - heuristic action
        # Ensure critical files exist.
        for relative in [
            "README.md",
            "langgraph.json",
            "pyproject.toml",
        ]:
            target = project_path / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")

    def _fix_directory_structure(self, project_path: Path) -> None:  # pragma: no cover - heuristic action
        # Ensure src directories exist
        for directory in [project_path / "src" / name for name in ("agent", "config", "llm")]:
            directory.mkdir(parents=True, exist_ok=True)

    def _update_configuration(self, project_path: Path) -> None:  # pragma: no cover - heuristic action
        langgraph_file = project_path / "langgraph.json"
        if langgraph_file.exists():
            self._update_langgraph_json_paths(project_path)


def code_fixer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    fixer = CodeFixer()
    return fixer.fix_project_issues(state)
