"""Guarantee generated modules import local shims for shared utilities."""

from __future__ import annotations

import ast
import logging
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

from asb.utils.fileops import atomic_write, ensure_dir

logger = logging.getLogger(__name__)

_MESSAGE_UTILS_SOURCE = """from __future__ import annotations

from typing import Any, List


def extract_last_message_content(messages: List[Any], default: str = "") -> str:
    if not messages:
        return default

    last_message = messages[-1]
    content = None
    if hasattr(last_message, "content"):
        content = getattr(last_message, "content", None)
    elif isinstance(last_message, dict):
        content = last_message.get("content")
    elif isinstance(last_message, str):
        content = last_message

    if content in (None, ""):
        return default
    return str(content)


def extract_user_messages_content(messages: List[Any]) -> List[str]:
    collected: List[str] = []
    for message in messages:
        role = None
        content = None
        if hasattr(message, "type"):
            role = getattr(message, "type", None)
        if hasattr(message, "role"):
            role = getattr(message, "role", role)
        if hasattr(message, "content"):
            content = getattr(message, "content", None)
        elif isinstance(message, dict):
            role = message.get("role", role)
            content = message.get("content")
        if isinstance(role, str) and role.lower() in {"human", "user"}:
            if content not in (None, ""):
                collected.append(str(content))
    return collected


def safe_message_access(message: Any, field: str, default: Any = "") -> Any:
    if hasattr(message, field):
        return getattr(message, field, default)
    if isinstance(message, dict):
        return message.get(field, default)
    return default


__all__ = [
    "extract_last_message_content",
    "extract_user_messages_content",
    "safe_message_access",
]
"""


def _determine_project_root(state: Dict[str, Any]) -> Path | None:
    scaffold = state.get("scaffold")
    if isinstance(scaffold, dict):
        path = scaffold.get("path")
        if path:
            return Path(path)
    candidate = state.get("project_root")
    if candidate:
        return Path(str(candidate))
    return None


def _iter_node_modules(agent_dir: Path) -> Iterator[Path]:
    for path in agent_dir.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        yield path


def _ensure_message_utils(agent_dir: Path) -> None:
    utils_dir = agent_dir / "utils"
    ensure_dir(utils_dir)
    init_path = utils_dir / "__init__.py"
    if not init_path.exists():
        atomic_write(init_path, "__all__ = []\n")
    target = utils_dir / "message_utils.py"
    if not target.exists() or target.read_text(encoding="utf-8") != _MESSAGE_UTILS_SOURCE:
        atomic_write(target, _MESSAGE_UTILS_SOURCE)


def _normalize_stub_target(agent_dir: Path, target: Path) -> Path:
    try:
        relative = target.relative_to(agent_dir)
    except ValueError:
        return target

    normalized_parts: list[str] = []
    for part in relative.parts:
        if not part:
            continue
        if normalized_parts and part == normalized_parts[-1]:
            continue
        normalized_parts.append(part)

    if not normalized_parts:
        return agent_dir

    return agent_dir.joinpath(*normalized_parts)


def _normalize_imports_text(source: str) -> tuple[str, bool]:
    updated = source.replace(
        "from asb.utils.message_utils import extract_last_message_content",
        "from .utils.message_utils import extract_last_message_content",
    )
    if (
        "extract_last_message_content" in updated
        and "from .utils.message_utils" not in updated
    ):
        lines = updated.splitlines()
        insert_at = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from __future__"):
                insert_at = index + 1
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                insert_at = index + 1
            else:
                break
        lines.insert(
            insert_at,
            "from .utils.message_utils import extract_last_message_content",
        )
        updated = "\n".join(lines) + ("\n" if updated.endswith("\n") else "")
    return updated, updated != source


def _ensure_init_file(directory: Path, created: set[Path], created_paths: list[str]) -> None:
    if directory in created:
        return
    init_path = directory / "__init__.py"
    if not init_path.exists():
        atomic_write(init_path, "__all__ = []\n")
        created_paths.append(str(init_path))
    created.add(directory)


def _ensure_package_structure(agent_dir: Path) -> list[str]:
    created: list[str] = []
    ensured: set[Path] = set()
    if not agent_dir.exists():
        return created

    _ensure_init_file(agent_dir, ensured, created)
    for module_path in agent_dir.rglob("*.py"):
        current = module_path.parent
        while True:
            _ensure_init_file(current, ensured, created)
            if current == agent_dir:
                break
            current = current.parent
    return created


_DEPENDENCY_KEYS: Sequence[str] = (
    "dependencies",
    "python_dependencies",
    "libraries",
    "packages",
    "imports",
)


def _normalize_dependency_name(value: str) -> str:
    candidate = value.split(";", 1)[0]
    candidate = re.split(r"[<>=!~]", candidate, 1)[0]
    candidate = candidate.replace("-", "_")
    candidate = candidate.strip()
    candidate = candidate.split()[0] if candidate else ""
    return candidate


def _collect_dependency_strings(value: object) -> set[str]:
    collected: set[str] = set()
    if isinstance(value, str):
        name = _normalize_dependency_name(value)
        if name:
            collected.add(name)
    elif isinstance(value, Mapping):
        for item in value.values():
            collected.update(_collect_dependency_strings(item))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            collected.update(_collect_dependency_strings(item))
    return collected


def _collect_plan_modules(state: Mapping[str, Any]) -> set[str]:
    plan = state.get("plan") if isinstance(state, Mapping) else None
    if not isinstance(plan, Mapping):
        return set()

    modules: set[str] = set()
    for key in _DEPENDENCY_KEYS:
        modules.update(_collect_dependency_strings(plan.get(key)))

    nodes = plan.get("nodes")
    if isinstance(nodes, Sequence):
        for node in nodes:
            if isinstance(node, Mapping):
                for key in _DEPENDENCY_KEYS:
                    modules.update(_collect_dependency_strings(node.get(key)))
    return modules


def _collect_requirements_modules(state: Mapping[str, Any]) -> set[str]:
    requirements = state.get("requirements") if isinstance(state, Mapping) else None
    if not isinstance(requirements, Mapping):
        return set()
    modules: set[str] = set()
    for key in _DEPENDENCY_KEYS:
        modules.update(_collect_dependency_strings(requirements.get(key)))
    return modules


def _collect_pyproject_modules(project_root: Path) -> set[str]:
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return set()
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError):
        return set()

    modules: set[str] = set()
    project = data.get("project")
    if isinstance(project, Mapping):
        dependencies = project.get("dependencies")
        if isinstance(dependencies, Sequence):
            for dep in dependencies:
                modules.update(_collect_dependency_strings(dep))
        optional = project.get("optional-dependencies")
        if isinstance(optional, Mapping):
            for extras in optional.values():
                modules.update(_collect_dependency_strings(extras))
    return modules


def _collect_project_packages(project_root: Path) -> set[str]:
    src_dir = project_root / "src"
    modules: set[str] = set()
    if not src_dir.exists():
        return modules
    for entry in src_dir.iterdir():
        if entry.is_dir():
            modules.add(entry.name)
        elif entry.suffix == ".py":
            modules.add(entry.stem)
    return modules


def _build_allowed_modules(state: Mapping[str, Any], project_root: Path) -> tuple[set[str], Dict[str, Any]]:
    stdlib_modules = set(sys.stdlib_module_names)
    stdlib_modules.update({"__future__", "builtins"})
    pyproject_modules = _collect_pyproject_modules(project_root)
    plan_modules = _collect_plan_modules(state)
    requirements_modules = _collect_requirements_modules(state)
    project_modules = _collect_project_packages(project_root)

    allowed = set(stdlib_modules)
    allowed.update(pyproject_modules)
    allowed.update(plan_modules)
    allowed.update(requirements_modules)
    allowed.update(project_modules)

    metadata = {
        "stdlib_count": len(stdlib_modules),
        "pyproject": sorted(pyproject_modules),
        "plan": sorted(plan_modules),
        "requirements": sorted(requirements_modules),
        "project_packages": sorted(project_modules),
    }
    return allowed, metadata


def _top_level_module(name: str) -> str:
    return name.split(".", 1)[0].strip()


def _resolve_relative_base(module_path: Path, level: int, agent_dir: Path) -> Path | None:
    if level <= 0:
        return module_path.parent
    base = module_path.parent
    steps = max(level - 1, 0)
    for _ in range(steps):
        base = base.parent
        if agent_dir not in base.parents and base != agent_dir:
            return None
    return base


def _register_module_stub(
    agent_dir: Path,
    module_path: Path,
    alias_names: Iterable[str],
    attribute_stubs: MutableMapping[Path, set[str]],
    module_presence: set[Path],
) -> None:
    module_path = _normalize_stub_target(agent_dir, module_path)
    alias_set = {name for name in alias_names if name and name != "*"}
    if module_path.exists():
        if alias_set:
            attribute_stubs[module_path].update(alias_set)
        return
    if module_path.suffix == ".py":
        if alias_set:
            attribute_stubs[module_path].update(alias_set)
        else:
            module_presence.add(module_path)
    else:
        module_presence.add(module_path)


def _render_stub_module(names: Iterable[str]) -> str:
    exported = sorted({name for name in names if name and name != "*"})
    lines = [
        "from __future__ import annotations",
        "",
        '"""Auto-generated stub to satisfy relative imports."""',
        "",
    ]
    if exported:
        exports = ", ".join(f"'{name}'" for name in exported)
        lines.append(f"__all__ = [{exports}]")
        lines.append("")
        for name in exported:
            lines.append(f"def {name}(*args: object, **kwargs: object) -> None:")
            lines.append("    \"\"\"Stub generated by the import resolver.\"\"\"")
            lines.append("    return None")
            lines.append("")
    else:
        lines.append("__all__: list[str] = []")
        lines.append("")
        lines.append("pass")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def import_resolver_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure generated nodes depend only on local utility shims."""

    working_state: Dict[str, Any] = dict(state or {})
    project_root = _determine_project_root(working_state)
    if project_root is None:
        return working_state

    agent_dir = project_root / "src" / "agent"
    if not agent_dir.exists():
        return working_state

    _ensure_message_utils(agent_dir)
    created_inits = _ensure_package_structure(agent_dir)

    allowed_modules, whitelist_meta = _build_allowed_modules(working_state, project_root)

    attribute_stubs: defaultdict[Path, set[str]] = defaultdict(set)
    module_presence: set[Path] = set()
    illegal_imports: list[Dict[str, Any]] = []
    normalized_modules: list[str] = []
    scanned_modules: list[str] = []
    parse_failures: list[str] = []

    for module in _iter_node_modules(agent_dir):
        try:
            source = module.read_text(encoding="utf-8")
        except OSError as exc:
            parse_failures.append(f"{module}: unable to read ({exc})")
            continue

        normalized_source, changed = _normalize_imports_text(source)
        if changed:
            atomic_write(module, normalized_source)
            normalized_modules.append(str(module))
        else:
            normalized_source = source

        try:
            tree = ast.parse(normalized_source, filename=str(module))
        except SyntaxError as exc:
            parse_failures.append(f"{module}: unable to parse ({exc})")
            continue

        scanned_modules.append(str(module))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name or ""
                    top_level = _top_level_module(module_name)
                    if not top_level or top_level in allowed_modules:
                        continue
                    illegal_imports.append(
                        {
                            "module": module_name,
                            "source": str(module),
                            "lineno": getattr(node, "lineno", None),
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    module_name = node.module or ""
                    top_level = _top_level_module(module_name)
                    if top_level and top_level not in allowed_modules:
                        illegal_imports.append(
                            {
                                "module": module_name,
                                "source": str(module),
                                "lineno": getattr(node, "lineno", None),
                            }
                        )
                    continue

                base = _resolve_relative_base(module, node.level, agent_dir)
                if base is None:
                    continue

                module_parts = node.module.split(".") if node.module else []
                alias_names = [alias.name for alias in node.names]

                if module_parts:
                    parent = base
                    for part in module_parts[:-1]:
                        parent = parent / part
                        ensure_dir(parent)
                    leaf = module_parts[-1]
                    package_dir = parent / leaf
                    module_file = package_dir.with_suffix(".py")

                    if package_dir.exists() and not module_file.exists():
                        ensure_dir(package_dir)
                        for alias_name in alias_names:
                            if alias_name == "*":
                                continue
                            target = package_dir / f"{alias_name}.py"
                            ensure_dir(target.parent)
                            _register_module_stub(
                                agent_dir,
                                target,
                                [],
                                attribute_stubs,
                                module_presence,
                            )
                    else:
                        ensure_dir(module_file.parent)
                        _register_module_stub(
                            agent_dir,
                            module_file,
                            alias_names,
                            attribute_stubs,
                            module_presence,
                        )
                else:
                    ensure_dir(base)
                    for alias_name in alias_names:
                        if alias_name == "*":
                            continue
                        target = base / f"{alias_name}.py"
                        ensure_dir(target.parent)
                        _register_module_stub(
                            agent_dir,
                            target,
                            [],
                            attribute_stubs,
                            module_presence,
                        )

    created_stub_modules: list[str] = []

    for path, names in sorted(attribute_stubs.items(), key=lambda item: str(item[0])):
        ensure_dir(path.parent)
        if path.exists():
            continue
        atomic_write(path, _render_stub_module(names))
        created_stub_modules.append(str(path))

    for path in sorted(module_presence, key=str):
        ensure_dir(path.parent)
        if path.exists():
            continue
        atomic_write(path, _render_stub_module([]))
        created_stub_modules.append(str(path))

    created_inits.extend(_ensure_package_structure(agent_dir))

    if illegal_imports:
        logger.error("Disallowed absolute imports detected: %s", illegal_imports)

    scratch = dict(working_state.get("scratch") or {})
    artifacts = scratch.setdefault("artifacts", {})
    artifacts["imports"] = {
        "message_utils": str(agent_dir / "utils" / "message_utils.py"),
        "stubs": created_stub_modules,
        "packages": created_inits,
    }

    resolver_log = {
        "scanned_modules": scanned_modules,
        "normalized_modules": normalized_modules,
        "created_stubs": created_stub_modules,
        "created_inits": created_inits,
        "illegal_imports": illegal_imports,
        "parse_failures": parse_failures,
        "whitelist": whitelist_meta,
    }
    scratch["import_resolver"] = resolver_log

    if illegal_imports:
        error_messages = [
            f"{entry['source']}: disallowed import '{entry['module']}'"
            for entry in illegal_imports
        ]
        working_errors = list(working_state.get("errors") or [])
        working_errors.extend(error_messages)
        working_state["errors"] = working_errors

    working_state["scratch"] = scratch
    return working_state


__all__ = ["import_resolver_node"]
