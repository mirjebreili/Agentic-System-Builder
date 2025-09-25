"""Comprehensive end-to-end automation for generated LangGraph projects.

This module orchestrates the full validation workflow for applications
produced by the Agentic System Builder.  It covers generation, structural
validation, server execution and API testing to ensure a high degree of
confidence before shipping generated projects.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import aiohttp


TEST_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "summarization_workflow",
        "user_prompt": "Create an agentic summarizer that takes input text and returns a summarized version",
        "expected_nodes": ["plan", "do", "finish"],
        "test_input": {
            "input_text": "This is a long article about artificial intelligence and machine learning technologies that are transforming various industries..."
        },
        "success_criteria": {
            "response_contains": ["summary", "key points", "conclusion"],
            "response_length_max": 500,
            "processing_time_max": 30,
        },
    },
    {
        "name": "chat_assistant",
        "user_prompt": "Build a helpful chat assistant that answers user questions",
        "expected_nodes": ["understand", "research", "respond"],
        "test_input": {"input_text": "What is the capital of France?"},
        "success_criteria": {
            "response_contains": ["Paris"],
            "response_length_min": 10,
            "processing_time_max": 20,
        },
    },
    {
        "name": "data_analysis_workflow",
        "user_prompt": "Create a data analysis agent that processes CSV files",
        "expected_nodes": ["load", "analyze", "report"],
        "test_input": {"input_text": "Analyze sales data trends"},
        "success_criteria": {
            "response_contains": ["analysis", "trends", "insights"],
            "response_length_min": 50,
            "processing_time_max": 45,
        },
    },
]


@dataclass
class AppGenerationResult:
    """Result of generating an application from the Agentic System Builder."""

    success: bool
    project_path: Optional[Path] = None
    generation_time: float = 0.0
    node_count: int = 0
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        if not self.success:
            return f"Generation failed: {self.error or 'unknown error'}"
        return f"Generated project at {self.project_path} ({self.node_count} nodes)"


@dataclass
class StructureValidationResult:
    """Result of validating the generated project structure."""

    success: bool
    issues: Sequence[str] = field(default_factory=list)
    files_validated: int = 0

    @property
    def summary(self) -> str:
        if self.success:
            return f"Validated {self.files_validated} files"
        return f"Validation failed with {len(self.issues)} issues"


@dataclass
class ServerStartResult:
    """Information about the LangGraph dev server startup."""

    success: bool
    server_url: Optional[str] = None
    process: Optional[asyncio.subprocess.Process] = None
    startup_time: float = 0.0
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        if self.success:
            return f"Server running at {self.server_url} in {self.startup_time:.2f}s"
        return f"Server failed: {self.error or 'unknown error'}"


@dataclass
class APITestResult:
    """Result of invoking the generated application's API."""

    success: bool
    response_content: str = ""
    processing_time: float = 0.0
    validation_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def summary(self) -> str:
        if self.success:
            return f"API OK in {self.processing_time:.2f}s"
        return f"API failed: {self.error or 'unknown error'}"


@dataclass
class TestResult:
    """Aggregate result for a complete scenario run."""

    success: bool
    summary: str = ""
    error: Optional[str] = None
    total_time: float = 0.0
    app_result: Optional[AppGenerationResult] = None
    structure_result: Optional[StructureValidationResult] = None
    server_result: Optional[ServerStartResult] = None
    api_result: Optional[APITestResult] = None
    detailed_issues: List[str] = field(default_factory=list)

    @classmethod
    def combine(
        cls,
        app_result: Optional[AppGenerationResult] = None,
        structure_result: Optional[StructureValidationResult] = None,
        server_result: Optional[ServerStartResult] = None,
        api_result: Optional[APITestResult] = None,
    ) -> "TestResult":
        results = [app_result, structure_result, server_result, api_result]
        success = all(r is None or getattr(r, "success", True) for r in results)

        errors: List[str] = []
        detailed_issues: List[str] = []

        if app_result and not app_result.success and app_result.error:
            errors.append(app_result.error)
        if structure_result and not structure_result.success:
            errors.extend(structure_result.issues)
            detailed_issues.extend(structure_result.issues)
        if server_result and not server_result.success and server_result.error:
            errors.append(server_result.error)
        if api_result and not api_result.success and api_result.error:
            errors.append(api_result.error)
            if api_result.validation_details and api_result.validation_details.get("issues"):
                detailed_issues.extend(api_result.validation_details["issues"])

        summary_parts: List[str] = []
        for r in results:
            if r is None:
                continue
            summary_attr = getattr(r, "summary", None)
            if summary_attr:
                summary_parts.append(summary_attr)

        total_time = 0.0
        if app_result:
            total_time += app_result.generation_time
        if server_result:
            total_time += server_result.startup_time
        if api_result:
            total_time += api_result.processing_time

        summary = " | ".join(summary_parts) if summary_parts else "No summary available"
        error = "; ".join(errors) if errors else None

        return cls(
            success=success,
            summary=summary,
            error=error,
            total_time=total_time,
            app_result=app_result,
            structure_result=structure_result,
            server_result=server_result,
            api_result=api_result,
            detailed_issues=detailed_issues,
        )


@dataclass
class E2ETestReport:
    """Comprehensive report for all E2E scenarios."""

    scenario_results: Sequence[Tuple[str, TestResult]]
    total_scenarios: int
    passed_scenarios: int
    total_execution_time: float
    timestamp: datetime

    @property
    def success_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.passed_scenarios / self.total_scenarios


class E2ETestRunner:
    """Runner that orchestrates the end-to-end validation workflow."""

    def __init__(self, base_projects_dir: Path, timeout_seconds: int = 60):
        self.base_projects_dir = base_projects_dir
        self.timeout = timeout_seconds
        self.active_servers: List[asyncio.subprocess.Process] = []
        self._scenario_context: Optional[Dict[str, Any]] = None

    async def run_full_e2e_test(self, scenario: Dict[str, Any]) -> TestResult:
        """Complete end-to-end test for a single scenario."""

        start_time = time.time()
        self._scenario_context = scenario

        app_result = await self.generate_application(scenario["user_prompt"])
        if not app_result.success:
            total_time = time.time() - start_time
            result = TestResult.combine(app_result)
            result.total_time = total_time
            return result

        structure_result = await self.validate_project_structure(app_result.project_path)
        if not structure_result.success:
            total_time = time.time() - start_time
            result = TestResult.combine(app_result, structure_result)
            result.total_time = total_time
            return result

        server_result = await self.start_langgraph_server(app_result.project_path)
        if not server_result.success:
            total_time = time.time() - start_time
            result = TestResult.combine(app_result, structure_result, server_result)
            result.total_time = total_time
            return result

        api_result = await self.test_api_functionality(
            server_result.server_url,
            scenario["test_input"],
            scenario["success_criteria"],
        )

        await self.cleanup_server(server_result.process)

        total_time = time.time() - start_time
        combined = TestResult.combine(app_result, structure_result, server_result, api_result)
        combined.total_time = total_time
        return combined

    async def generate_application(self, user_prompt: str) -> AppGenerationResult:
        """Test the core application generation pipeline."""

        state = {
            "input_text": user_prompt,
            "goal": user_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            from asb.agent.graph import graph

            start_time = time.time()
            result = await graph.ainvoke(
                state,
                config={"configurable": {"thread_id": f"e2e-test-{uuid4()}"}},
            )
            generation_time = time.time() - start_time

            scaffold = result.get("scaffold", {}) if isinstance(result, dict) else {}
            if not scaffold.get("ok", False):
                return AppGenerationResult(
                    success=False,
                    error="Application generation failed",
                )

            project_path = Path(scaffold["path"])
            architecture = result.get("architecture", {}) if isinstance(result, dict) else {}
            nodes = architecture.get("nodes", []) if isinstance(architecture, dict) else []

            return AppGenerationResult(
                success=True,
                project_path=project_path,
                generation_time=generation_time,
                node_count=len(nodes),
            )
        except Exception as exc:  # pragma: no cover - defensive programming
            return AppGenerationResult(success=False, error=str(exc))

    async def validate_project_structure(self, project_path: Optional[Path]) -> StructureValidationResult:
        """Validate generated project has correct structure and non-empty files."""

        if not project_path:
            return StructureValidationResult(
                success=False,
                issues=["Project path missing"],
                files_validated=0,
            )

        validation_checks: List[Tuple[Path, str]] = [
            (project_path / "src/agent/graph.py", "graph module"),
            (project_path / "src/agent/executor.py", "executor module"),
            (project_path / "src/agent/state.py", "state schema"),
            (project_path / "langgraph.json", "langgraph config"),
            (project_path / "pyproject.toml", "project config"),
        ]

        for node in self.expected_nodes_for_project(project_path):
            validation_checks.append((project_path / f"src/agent/{node}.py", f"{node} node"))

        issues: List[str] = []

        for file_path, description in validation_checks:
            if not file_path.exists():
                issues.append(f"Missing {description}: {file_path}")
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                issues.append(f"Failed reading {description}: {file_path} ({exc})")
                continue

            if len(content.strip()) < 50:
                issues.append(f"Empty or minimal {description}: {file_path}")
                continue

            if file_path.name == "executor.py" and "NODE_IMPLEMENTATIONS = []" in content:
                issues.append(f"Empty NODE_IMPLEMENTATIONS in {file_path}")
            elif file_path.name == "graph.py" and '"error": "No architecture plan provided"' in content:
                issues.append(f"Empty architecture plan in {file_path}")

        return StructureValidationResult(
            success=len(issues) == 0,
            issues=issues,
            files_validated=len(validation_checks),
        )

    async def start_langgraph_server(self, project_path: Optional[Path]) -> ServerStartResult:
        """Start LangGraph dev server and wait for it to be ready."""

        if not project_path:
            return ServerStartResult(success=False, error="Project path missing")

        env = os.environ.copy()
        env["LANGSMITH_TRACING"] = "false"

        try:
            process = await asyncio.create_subprocess_exec(
                "langgraph",
                "dev",
                "--port",
                "0",
                cwd=project_path,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return ServerStartResult(success=False, error=f"langgraph CLI missing: {exc}")

        start_time = time.time()
        server_url: Optional[str] = None

        while time.time() - start_time < self.timeout:
            if process.returncode is not None:
                stderr = await process.stderr.read()
                return ServerStartResult(
                    success=False,
                    error=f"Server failed to start: {stderr.decode().strip()}",
                )

            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if not line:
                continue

            line_str = line.decode().strip()
            match = re.search(r"http://localhost:(\d+)", line_str)
            if match:
                server_url = f"http://localhost:{match.group(1)}"
                break

        if not server_url:
            process.kill()
            await process.wait()
            return ServerStartResult(success=False, error="Server did not start within timeout")

        if not await self.verify_server_health(server_url):
            process.kill()
            await process.wait()
            return ServerStartResult(success=False, error="Server health check failed")

        self.active_servers.append(process)
        return ServerStartResult(
            success=True,
            server_url=server_url,
            process=process,
            startup_time=time.time() - start_time,
        )

    async def verify_server_health(self, server_url: str) -> bool:
        """Check if LangGraph server is responding correctly."""

        health_endpoints = [
            f"{server_url}/",
            f"{server_url}/docs",
            f"{server_url}/graphs",
        ]

        async with aiohttp.ClientSession() as session:
            for endpoint in health_endpoints:
                try:
                    async with session.get(endpoint, timeout=5) as response:
                        if response.status == 200:
                            return True
                except Exception:
                    continue

        return False

    async def test_api_functionality(
        self,
        server_url: Optional[str],
        test_input: Dict[str, Any],
        success_criteria: Dict[str, Any],
    ) -> APITestResult:
        """Test the generated application's API functionality."""

        if not server_url:
            return APITestResult(success=False, error="Server URL missing")

        payload = {
            "input": test_input,
            "config": {"configurable": {"thread_id": f"test-{uuid4()}"}},
            "stream": False,
        }

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                timeout = success_criteria.get("processing_time_max", 60)
                async with session.post(
                    f"{server_url}/graphs/agent/invoke",
                    json=payload,
                    timeout=timeout,
                ) as response:
                    processing_time = time.time() - start_time

                    if response.status != 200:
                        return APITestResult(
                            success=False,
                            error=f"API returned status {response.status}",
                            processing_time=processing_time,
                        )

                    response_data = await response.json()
                    response_content = self.extract_response_content(response_data)
                    validation_results = self.validate_response_criteria(
                        response_content,
                        success_criteria,
                        processing_time,
                    )

                    return APITestResult(
                        success=validation_results["success"],
                        response_content=response_content,
                        processing_time=processing_time,
                        validation_details=validation_results,
                        error=None if validation_results["success"] else validation_results["error"],
                    )

        except Exception as exc:  # pragma: no cover - network/process issues
            return APITestResult(
                success=False,
                error=str(exc),
                processing_time=time.time() - start_time,
            )

    def validate_response_criteria(
        self,
        response_content: str,
        criteria: Dict[str, Any],
        processing_time: float,
    ) -> Dict[str, Any]:
        """Validate response meets success criteria."""

        issues: List[str] = []

        if "response_contains" in criteria:
            for required_text in criteria["response_contains"]:
                if required_text.lower() not in response_content.lower():
                    issues.append(f"Response missing required content: '{required_text}'")

        if "response_length_min" in criteria:
            min_len = criteria["response_length_min"]
            if len(response_content) < min_len:
                issues.append(
                    f"Response too short: {len(response_content)} < {min_len}"
                )

        if "response_length_max" in criteria:
            max_len = criteria["response_length_max"]
            if len(response_content) > max_len:
                issues.append(
                    f"Response too long: {len(response_content)} > {max_len}"
                )

        if "processing_time_max" in criteria:
            max_time = criteria["processing_time_max"]
            if processing_time > max_time:
                issues.append(
                    f"Processing too slow: {processing_time:.2f}s > {max_time}s"
                )

        return {
            "success": len(issues) == 0,
            "issues": issues,
            "error": "; ".join(issues) if issues else None,
        }

    def expected_nodes_for_project(self, project_path: Path) -> Iterable[str]:
        """Determine expected nodes for the given project.

        Prefer scenario context but fallback to inspecting generated artifacts.
        """

        if self._scenario_context and self._scenario_context.get("expected_nodes"):
            return list(self._scenario_context["expected_nodes"])

        nodes: List[str] = []
        langgraph_file = project_path / "langgraph.json"
        if langgraph_file.exists():
            try:
                config = json.loads(langgraph_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}
            nodes = config.get("expected_nodes", []) if isinstance(config, dict) else []

        if not nodes:
            agent_dir = project_path / "src/agent"
            if agent_dir.exists():
                nodes = [
                    p.stem
                    for p in agent_dir.glob("*.py")
                    if p.name not in {"graph.py", "executor.py", "state.py", "__init__.py"}
                ]

        return nodes

    def extract_response_content(self, response_data: Any) -> str:
        """Attempt to normalise API responses into a string."""

        if response_data is None:
            return ""

        if isinstance(response_data, str):
            return response_data

        if isinstance(response_data, dict):
            for key in ("output", "result", "data", "response", "content"):
                if key in response_data:
                    return self.extract_response_content(response_data[key])

            if "messages" in response_data:
                messages = response_data["messages"]
                if isinstance(messages, list):
                    combined = "\n".join(
                        self.extract_response_content(m) for m in messages
                    )
                    if combined:
                        return combined

            return json.dumps(response_data)

        if isinstance(response_data, list):
            combined = "\n".join(self.extract_response_content(item) for item in response_data)
            if combined:
                return combined

        return str(response_data)

    async def cleanup_server(self, process: Optional[asyncio.subprocess.Process]) -> None:
        """Terminate the given server process and remove it from tracking."""

        if not process:
            return

        if process in self.active_servers:
            self.active_servers.remove(process)

        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

    async def cleanup(self) -> None:
        """Terminate all tracked server processes."""

        await asyncio.gather(*(self.cleanup_server(proc) for proc in list(self.active_servers)))

    async def run_all_e2e_tests(self) -> E2ETestReport:
        """Run all test scenarios and generate comprehensive report."""

        results: List[Tuple[str, TestResult]] = []
        start_time = time.time()

        for scenario in TEST_SCENARIOS:
            print(f"🧪 Running E2E test: {scenario['name']}")

            try:
                result = await self.run_full_e2e_test(scenario)
                results.append((scenario["name"], result))

                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"   {status} - {result.summary}")

            except Exception as exc:  # pragma: no cover - orchestrator failure
                error_result = TestResult(success=False, error=str(exc))
                results.append((scenario["name"], error_result))
                print(f"   ❌ ERROR - {exc}")

        total_time = time.time() - start_time

        report = E2ETestReport(
            scenario_results=results,
            total_scenarios=len(TEST_SCENARIOS),
            passed_scenarios=sum(1 for _, result in results if result.success),
            total_execution_time=total_time,
            timestamp=datetime.utcnow(),
        )

        return report

    def generate_test_report(self, report: E2ETestReport) -> str:
        """Generate human-readable test report."""

        lines = [
            "🧪 END-TO-END TESTING REPORT",
            "=" * 50,
            "",
            f"📅 Timestamp: {report.timestamp.isoformat()}",
            f"⏱️  Total Time: {report.total_execution_time:.2f}s",
            f"📊 Results: {report.passed_scenarios}/{report.total_scenarios} scenarios passed",
            f"🎯 Success Rate: {report.success_rate:.1%}",
            "",
            "📋 SCENARIO DETAILS:",
        ]

        for scenario_name, result in report.scenario_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            lines.extend(
                [
                    "",
                    f"🔍 {scenario_name.upper()}",
                    f"   Status: {status}",
                    f"   Duration: {result.total_time:.2f}s",
                ]
            )

            if result.success and result.api_result:
                lines.append(
                    f"   API Response: {result.api_result.processing_time:.2f}s"
                )
            else:
                lines.append(f"   Error: {result.error or 'Unknown error'}")
                if result.detailed_issues:
                    for issue in list(result.detailed_issues)[:3]:
                        lines.append(f"     • {issue}")

        return "\n".join(lines)


__all__ = [
    "TEST_SCENARIOS",
    "AppGenerationResult",
    "StructureValidationResult",
    "ServerStartResult",
    "APITestResult",
    "TestResult",
    "E2ETestReport",
    "E2ETestRunner",
]

