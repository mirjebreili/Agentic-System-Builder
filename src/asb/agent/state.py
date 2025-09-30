"""Updated state schema to include package discovery fields."""
from __future__ import annotations
from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig


class AppState(TypedDict):
    # Original fields
    input: str
    plan: Optional[str]
    confidence: Optional[float] 
    replan: Optional[bool]
    
    # Package discovery fields
    package_plan: Optional[Dict[str, Any]]
    search_queries: Optional[List[List[str]]]
    npm_candidates: Optional[List[Dict[str, Any]]]
    pypi_candidates: Optional[List[Dict[str, Any]]]
    total_candidates: Optional[int]
    ranked_packages: Optional[List[Dict[str, Any]]]
    capability_groups: Optional[Dict[str, List[Dict[str, Any]]]]
    ranking_summary: Optional[Dict[str, Any]]
    integration_code: Optional[str]
    deployment_guidance: Optional[str] 
    selected_package_summary: Optional[List[Dict[str, Any]]]
    integration_stats: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    replan_reason: Optional[str]
    final_response: Optional[str]
