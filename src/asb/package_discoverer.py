"""
Multi-ecosystem package discovery using npm and PyPI APIs with enhanced metadata collection.
"""
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import httpx
from asb.llm import get_llm


# Utility functions
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([^/\s]+)/([^/\s]+)")

def extract_github_repo(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = GITHUB_RE.search(url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).rstrip(".git")
    return f"github.com/{owner}/{repo}"


async def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 20.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None


# npm discovery functions
async def npm_search(query: str, size: int = 25, from_: int = 0) -> List[Dict[str, Any]]:
    url = "https://registry.npmjs.org/-/v1/search"
    params = {"text": query, "size": size, "from": from_}
    data = await http_get_json(url, params=params)
    return data.get("objects", []) if data else []


async def npm_downloads_last_week(pkg: str) -> Optional[int]:
    try:
        data = await http_get_json(f"https://api.npmjs.org/downloads/point/last-week/{pkg}")
        return int(data.get("downloads", 0)) if data else 0
    except Exception:
        return 0


# PyPI discovery functions
async def pypi_project_json(name: str) -> Optional[Dict[str, Any]]:
    try:
        return await http_get_json(f"https://pypi.org/pypi/{name}/json")
    except Exception:
        return None


async def pypi_search_via_llm(query: str, llm) -> List[str]:
    """Use LLM to suggest likely PyPI package names for a query."""
    prompt = f"""
List 15 likely Python package names that could help with this requirement.
Focus on well-known, established packages. Return only a JSON array of strings.

Requirement: {query}

Example format: ["requests", "flask", "pandas"]
"""
    
    response = await llm.ainvoke(prompt)
    text = getattr(response, "content", str(response))
    
    try:
        packages = json.loads(text)
        if isinstance(packages, list):
            return [re.sub(r"[^a-zA-Z0-9_\-\.]", "", p) for p in packages if p]
    except:
        pass
    
    # Fallback: extract package-like names from text
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_\-]*\b', text.lower())
    return words[:15]


# OpenSSF Scorecard integration
async def get_scorecard_data(github_project: str) -> Optional[Dict[str, Any]]:
    if not github_project:
        return None
    try:
        url = f"https://api.securityscorecards.dev/projects/{github_project}"
        return await http_get_json(url)
    except Exception:
        return None


async def discover_packages_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Discover relevant packages from npm and PyPI based on the package plan."""
    llm = get_llm()
    plan = state.get("package_plan", {})
    capabilities = plan.get("capabilities", [])
    
    npm_candidates = []
    pypi_candidates = []
    
    # Process each capability
    for capability in capabilities:
        search_keywords = capability.get("search_keywords", [])
        ecosystem_priority = capability.get("ecosystem_priority", ["pypi", "npm"])
        
        # Search in prioritized ecosystems
        for keyword in search_keywords[:3]:  # Limit to top 3 keywords
            
            # npm search
            if "npm" in ecosystem_priority:
                npm_results = await npm_search(keyword, size=10)
                for obj in npm_results:
                    pkg = obj.get("package", {})
                    name = pkg.get("name")
                    if not name:
                        continue
                    
                    downloads = await npm_downloads_last_week(name)
                    score = obj.get("score", {}) or {}
                    detail = score.get("detail", {}) or {}
                    links = pkg.get("links", {}) or {}
                    repo_url = links.get("repository") or links.get("homepage")
                    
                    npm_candidates.append({
                        "capability": capability["name"],
                        "ecosystem": "npm",
                        "name": name,
                        "version": pkg.get("version"),
                        "description": pkg.get("description"),
                        "keywords": pkg.get("keywords", []),
                        "date": pkg.get("date"),
                        "quality": detail.get("quality"),
                        "popularity": detail.get("popularity"), 
                        "maintenance": detail.get("maintenance"),
                        "downloads_week": downloads or 0,
                        "repo": repo_url,
                        "license": pkg.get("license"),
                    })
            
            # PyPI search via LLM suggestions
            if "pypi" in ecosystem_priority:
                pypi_suggestions = await pypi_search_via_llm(f"{keyword} {capability['name']}", llm)
                
                # Verify suggestions by checking if packages exist
                for suggestion in pypi_suggestions[:8]:  # Limit verification calls
                    if len(suggestion) < 2:
                        continue
                        
                    meta = await pypi_project_json(suggestion)
                    if not meta:
                        continue
                    
                    info = meta.get("info", {}) or {}
                    urls = info.get("project_urls", {}) or {}
                    homepage = info.get("home_page")
                    repo_url = urls.get("Source") or urls.get("Homepage") or urls.get("Repository") or homepage
                    
                    pypi_candidates.append({
                        "capability": capability["name"],
                        "ecosystem": "pypi",
                        "name": info.get("name") or suggestion,
                        "version": info.get("version"),
                        "summary": info.get("summary"),
                        "description": info.get("description"),
                        "keywords": info.get("keywords", "").split(",") if info.get("keywords") else [],
                        "license": info.get("license"),
                        "repo": repo_url,
                        "author": info.get("author"),
                        "author_email": info.get("author_email"),
                        "classifiers": info.get("classifiers", []),
                        "requires_dist": info.get("requires_dist", []),
                        "requires_python": info.get("requires_python"),
                        "releases": list((meta.get("releases") or {}).keys())[-5:],  # Last 5 versions
                    })
    
    return {
        "npm_candidates": npm_candidates,
        "pypi_candidates": pypi_candidates,
        "total_candidates": len(npm_candidates) + len(pypi_candidates)
    }
