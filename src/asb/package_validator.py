"""
Validate package security and compatibility before final recommendation.
"""
import asyncio
from typing import Dict, Any, List, Tuple


async def validate_packages_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive validation of selected packages."""
    
    ranked_packages = state.get("ranked_packages", [])
    selected_summary = state.get("selected_package_summary", [])
    
    validation_results = {
        "overall_status": "passed",
        "security_issues": [],
        "compatibility_warnings": [],
        "recommendations": [],
        "risk_assessment": "low"
    }
    
    high_risk_count = 0
    medium_risk_count = 0
    
    # Security validation
    for pkg in selected_summary[:10]:  # Validate top 10
        pkg_name = pkg.get("name")
        security_score = pkg.get("security_score")
        ecosystem = pkg.get("ecosystem")
        
        # Check OpenSSF Scorecard score
        if security_score is not None:
            if security_score < 3.0:
                validation_results["security_issues"].append(
                    f"LOW security score for {ecosystem}:{pkg_name} (score: {security_score:.1f})"
                )
                high_risk_count += 1
            elif security_score < 5.0:
                validation_results["security_issues"].append(
                    f"MEDIUM security score for {ecosystem}:{pkg_name} (score: {security_score:.1f})"
                )
                medium_risk_count += 1
        
        # Check maintenance status for npm packages
        if ecosystem == "npm":
            maintenance = next((p.get("maintenance") for p in ranked_packages if p.get("name") == pkg_name), None)
            if maintenance is not None and maintenance < 0.3:
                validation_results["compatibility_warnings"].append(
                    f"Low maintenance score for npm:{pkg_name} (score: {maintenance:.2f})"
                )
                medium_risk_count += 1
    
    # License compatibility check
    license_issues = []
    licenses = [pkg.get("license") for pkg in selected_summary if pkg.get("license")]
    
    restrictive_licenses = [l for l in licenses if l and any(term in l.lower() for term in ["gpl", "agpl", "copyleft"])]
    if restrictive_licenses:
        validation_results["compatibility_warnings"].append(
            f"Restrictive licenses detected: {', '.join(set(restrictive_licenses))}"
        )
    
    # Ecosystem balance check
    npm_count = len([p for p in selected_summary if p.get("ecosystem") == "npm"])
    pypi_count = len([p for p in selected_summary if p.get("ecosystem") == "pypi"])
    
    if npm_count > 0 and pypi_count > 0:
        validation_results["recommendations"].append(
            f"Mixed ecosystem solution ({npm_count} npm, {pypi_count} PyPI packages) - consider containerization"
        )
    
    # Risk assessment
    if high_risk_count > 2:
        validation_results["risk_assessment"] = "high"
        validation_results["overall_status"] = "failed"
    elif high_risk_count > 0 or medium_risk_count > 3:
        validation_results["risk_assessment"] = "medium"
        validation_results["overall_status"] = "warning"
    
    # Generate recommendations
    if validation_results["risk_assessment"] in ["medium", "high"]:
        validation_results["recommendations"].extend([
            "Consider implementing additional security monitoring",
            "Set up dependency update automation (Dependabot/Renovate)",
            "Run regular security audits (npm audit, pip-audit)",
        ])
    
    validation_results["recommendations"].append("Monitor package health with OpenSSF Scorecard")
    
    return {"validation_results": validation_results}


def should_replan_packages(state: Dict[str, Any]) -> str:
    """Decide whether to replan based on validation results."""
    validation = state.get("validation_results", {})
    status = validation.get("overall_status", "passed")
    
    return "True" if status == "failed" else "False"


async def replan_or_finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Either replan with stricter criteria or finalize the package solution."""
    
    validation = state.get("validation_results", {})
    status = validation.get("overall_status", "passed")
    
    if status == "failed":
        # Trigger replanning with stricter security requirements
        original_plan = state.get("package_plan", {})
        capabilities = original_plan.get("capabilities", [])
        
        # Add security constraints to each capability
        for capability in capabilities:
            capability["security_requirements"] = "high"
            capability["min_scorecard_score"] = 5.0
            
        return {
            "package_plan": {
                **original_plan,
                "capabilities": capabilities,
                "security_mode": "strict"
            },
            "replan_reason": "Failed security validation - applying stricter criteria"
        }
    
    # Finalize solution
    integration_code = state.get("integration_code", "")
    deployment_guidance = state.get("deployment_guidance", "")
    package_summary = state.get("selected_package_summary", [])
    integration_stats = state.get("integration_stats", {})
    
    final_report = f"""# Package Discovery Solution

## Overview
Successfully discovered and integrated {integration_stats.get('total_packages', 0)} packages across {integration_stats.get('npm_packages', 0)} npm and {integration_stats.get('pypi_packages', 0)} PyPI ecosystems.

## Validation Status
- **Overall Status**: {validation.get("overall_status", "unknown").upper()}
- **Risk Assessment**: {validation.get("risk_assessment", "unknown").upper()}
- **Security Issues**: {len(validation.get("security_issues", []))}
- **Compatibility Warnings**: {len(validation.get("compatibility_warnings", []))}

## Selected Packages
{chr(10).join([f"- **{p['ecosystem']}:{p['name']}** (v{p.get('version', 'latest')}) - {p.get('description', 'No description')[:100]}..." for p in package_summary[:10]])}

## Integration Solution
{integration_code}

## Deployment & Operations
{deployment_guidance}

## Validation Summary
### Security Issues:
{chr(10).join([f"- {issue}" for issue in validation.get("security_issues", [])]) or "None detected"}

### Recommendations:
{chr(10).join([f"- {rec}" for rec in validation.get("recommendations", [])]) or "Standard deployment practices"}

## Next Steps
1. Review the generated integration code
2. Set up the development environment as instructed
3. Run initial tests and validation
4. Implement monitoring and security measures
5. Deploy following the provided guidelines
"""
    
    return {"final_response": final_report}
