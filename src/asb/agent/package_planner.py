"""
Package-focused planner that creates implementation plans for package discovery and integration.
"""
from typing import Dict, Any, List
import json
from langchain_core.runnables import Runnable
from asb.llm import get_llm


async def package_planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a detailed plan for solving the problem using discoverable packages."""
    llm: Runnable = get_llm()
    
    prompt = f"""
You are a senior solutions architect specializing in package-driven development.
Given the objective below, create a detailed implementation plan that focuses on:
1. Breaking down the problem into capabilities that can be fulfilled by existing packages
2. Identifying the most likely package ecosystems (npm vs PyPI) for each capability
3. Creating a step-by-step integration strategy

Output your plan as a JSON object with this structure:
{{
    "capabilities": [
        {{
            "name": "capability_name",
            "description": "what this capability does",
            "ecosystem_priority": ["npm", "pypi"] or ["pypi", "npm"],
            "search_keywords": ["keyword1", "keyword2", ...],
            "integration_complexity": "low|medium|high"
        }}
    ],
    "architecture_approach": "microservices|monolithic|hybrid",
    "primary_language": "javascript|python|mixed",
    "integration_strategy": "description of how components will work together"
}}

Objective: {state["input"]}
"""
    
    response = await llm.ainvoke(prompt)
    text = getattr(response, "content", str(response))
    
    try:
        plan_data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback to simple capability extraction
        capabilities = []
        lines = text.strip().split('\n')
        for line in lines:
            if line.strip() and not line.startswith('#'):
                capabilities.append({
                    "name": line.strip(),
                    "description": line.strip(),
                    "ecosystem_priority": ["pypi", "npm"],
                    "search_keywords": [line.strip().lower()],
                    "integration_complexity": "medium"
                })
        
        plan_data = {
            "capabilities": capabilities,
            "architecture_approach": "hybrid",
            "primary_language": "python",
            "integration_strategy": "Package-based microservice composition"
        }
    
    return {
        "package_plan": plan_data,
        "search_queries": [cap["search_keywords"] for cap in plan_data["capabilities"]]
    }
