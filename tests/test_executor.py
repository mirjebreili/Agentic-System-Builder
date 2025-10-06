import conftest
import prompt2graph.llm.client as llm_client

from src.agent.executor import execute_deep


def test_get_chat_model_returns_fake_instance():
    llm = llm_client.get_chat_model()
    assert isinstance(llm, conftest.FakeChatModel)
    # invoking without special system message should return plan JSON
    content = llm.invoke([]).content
    assert "Test goal" in content


def test_executor_finishes():
    demo = {
        "plan": {
            "goal": "demo",
            "capabilities": [
                {
                    "name": "automation",
                    "description": "Automate workflow execution steps.",
                    "ecosystem_priority": ["pypi", "npm"],
                    "search_terms": ["prefect", "airflow", "bullmq"],
                    "complexity": "medium",
                    "required": True,
                },
                {
                    "name": "reporting",
                    "description": "Summarize output for the user.",
                    "ecosystem_priority": ["npm", "pypi"],
                    "search_terms": ["markdown", "rich", "notistack"],
                    "complexity": "low",
                    "required": True,
                },
            ],
            "architecture_approach": "hybrid",
            "primary_language": "mixed",
            "integration_strategy": "Combine Python orchestration with JS notification packages.",
            "confidence": 0.5,
        },
        "messages": [],
        "flags": {"more_steps": True, "steps_done": False},
    }
    out = execute_deep(demo)
    assert any("[execution]" in m.get("content", "") for m in out["messages"])

