"""
Generate integration code and architecture for selected packages.
"""
import json
from typing import Dict, Any, List
from asb.llm import get_llm


async def integrate_packages_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate integration code and architecture for top-ranked packages."""
    
    llm = get_llm()
    ranked_packages = state.get("ranked_packages", [])
    package_plan = state.get("package_plan", {})
    original_input = state.get("input", "")
    
    if not ranked_packages:
        return {"integration_code": "No packages found for integration."}
    
    # Prepare package summary for LLM
    package_summary = []
    npm_packages = []
    pypi_packages = []
    
    for pkg in ranked_packages[:12]:  # Top 12 packages
        ecosystem = pkg.get("ecosystem")
        summary_item = {
            "ecosystem": ecosystem,
            "name": pkg.get("name"),
            "capability": pkg.get("capability"),
            "description": pkg.get("description") or pkg.get("summary"),
            "version": pkg.get("version"),
            "composite_score": round(pkg.get("composite_score", 0), 3),
            "security_score": pkg.get("security_score"),
            "repo": pkg.get("repo"),
        }
        package_summary.append(summary_item)
        
        if ecosystem == "npm":
            npm_packages.append(pkg.get("name"))
        else:
            pypi_packages.append(pkg.get("name"))
    
    # Determine primary architecture
    architecture_approach = package_plan.get("architecture_approach", "hybrid")
    primary_language = package_plan.get("primary_language", "mixed")
    
    # Generate integration strategy
    integration_prompt = f"""
You are a senior platform engineer tasked with creating a working solution using the best packages discovered.

ORIGINAL REQUIREMENT:
{original_input}

PACKAGE PLAN:
{json.dumps(package_plan, indent=2)}

SELECTED PACKAGES (ranked by quality, security, and relevance):
{json.dumps(package_summary, indent=2)}

Create a comprehensive integration solution that includes:

1. **Architecture Decision**: Choose the optimal approach based on the packages and requirements
2. **Package Configuration**: Generate necessary package.json (for npm) and/or requirements.txt/pyproject.toml (for Python)
3. **Integration Code**: Create a working implementation that connects the selected packages
4. **Configuration Files**: Any config files needed (docker-compose.yml, .env template, etc.)
5. **Usage Instructions**: Clear setup and usage instructions

GUIDELINES:
- Prioritize packages with higher composite_score and security_score
- Create a production-ready, maintainable solution
- Include error handling and logging
- Focus on the top 6-8 packages that best fulfill the requirements
- If both npm and Python packages are selected, create a microservices architecture or use appropriate bridge technologies

Output your response as structured sections with clear markdown headers and code blocks.
"""
    
    response = await llm.ainvoke(integration_prompt)
    integration_text = getattr(response, "content", str(response))
    
    # Generate deployment guidance
    deployment_prompt = f"""
Based on the integration solution created, provide specific deployment and testing guidance:

SELECTED PACKAGES: {', '.join([p['name'] for p in package_summary[:8]])}
PRIMARY ARCHITECTURE: {architecture_approach}

Generate:
1. **Deployment Instructions**: Step-by-step deployment guide
2. **Testing Strategy**: How to test the integrated solution
3. **Monitoring Setup**: Key metrics and health checks
4. **Security Considerations**: Security best practices for the chosen packages
5. **Scaling Guidelines**: How to scale this solution

Keep it concise and actionable.
"""
    
    deployment_response = await llm.ainvoke(deployment_prompt)
    deployment_text = getattr(deployment_response, "content", str(deployment_response))
    
    return {
        "integration_code": integration_text,
        "deployment_guidance": deployment_text,
        "selected_package_summary": package_summary,
        "integration_stats": {
            "total_packages": len(package_summary),
            "npm_packages": len(npm_packages),
            "pypi_packages": len(pypi_packages),
            "primary_language": primary_language,
            "architecture": architecture_approach,
        }
    }
