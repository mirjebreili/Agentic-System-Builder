import json

def call_llm(prompt: str, vars: dict) -> str:
    """
    Stubbed LLM call that returns a fixed JSON string.
    """
    # This response should be what the planner_node expects to receive from the LLM.
    # Based on the final output, it seems the LLM should propose a plan.
    response = {
        "candidates": [
            {
                "plan": ["HttpBasedAtlasReadByKey", "membasedAtlasKeyStreamAggregator"],
                "rationale": "placeholder"
            }
        ]
    }
    return json.dumps(response)