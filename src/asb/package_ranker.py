"""
Advanced package ranking using multiple signals including security scores.
"""
import asyncio
import math
from typing import Dict, Any, List, Optional
from .package_discoverer import get_scorecard_data, extract_github_repo


def z_score_normalize(values: List[Optional[float]], target_value: Optional[float]) -> float:
    """Calculate z-score for normalization."""
    valid_values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not valid_values or not isinstance(target_value, (int, float)) or math.isnan(target_value):
        return 0.0
    
    mean = sum(valid_values) / len(valid_values)
    if len(valid_values) == 1:
        return 0.0
    
    variance = sum((v - mean) ** 2 for v in valid_values) / (len(valid_values) - 1)
    std_dev = math.sqrt(max(variance, 1e-9))
    
    return (target_value - mean) / std_dev


async def enhance_with_security_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enhance candidates with OpenSSF Scorecard security data."""
    
    # Collect GitHub repos and fetch scorecards in parallel
    scorecard_tasks = []
    for candidate in candidates:
        github_repo = extract_github_repo(candidate.get("repo"))
        if github_repo:
            scorecard_tasks.append(get_scorecard_data(github_repo))
        else:
            scorecard_tasks.append(asyncio.sleep(0, result=None))
    
    scorecards = await asyncio.gather(*scorecard_tasks, return_exceptions=False)
    
    # Attach scorecard data to candidates
    for candidate, scorecard in zip(candidates, scorecards):
        candidate["scorecard"] = scorecard or {}
        candidate["security_score"] = scorecard.get("score") if scorecard else None
    
    return candidates


async def rank_packages_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Rank discovered packages using multiple quality and security signals."""
    
    npm_candidates = state.get("npm_candidates", [])
    pypi_candidates = state.get("pypi_candidates", [])
    all_candidates = npm_candidates + pypi_candidates
    
    if not all_candidates:
        return {"ranked_packages": []}
    
    # Enhance with security scores
    all_candidates = await enhance_with_security_scores(all_candidates)
    
    # Collect metrics for normalization
    npm_downloads = [c.get("downloads_week") for c in npm_candidates]
    npm_quality = [c.get("quality") for c in npm_candidates] 
    npm_popularity = [c.get("popularity") for c in npm_candidates]
    npm_maintenance = [c.get("maintenance") for c in npm_candidates]
    
    security_scores = [c.get("security_score") for c in all_candidates]
    
    # Calculate composite scores
    ranked_candidates = []
    for candidate in all_candidates:
        ecosystem = candidate.get("ecosystem")
        
        # Base quality signals (ecosystem-specific)
        if ecosystem == "npm":
            quality_score = z_score_normalize(npm_quality, candidate.get("quality"))
            popularity_score = z_score_normalize(npm_popularity, candidate.get("popularity"))  
            maintenance_score = z_score_normalize(npm_maintenance, candidate.get("maintenance"))
            download_score = z_score_normalize(npm_downloads, candidate.get("downloads_week"))
        else:  # PyPI
            quality_score = 0.0
            popularity_score = 0.0 
            maintenance_score = 0.0
            download_score = 0.0
        
        # Security score (universal)
        security_z_score = z_score_normalize(security_scores, candidate.get("security_score"))
        
        # License scoring (prefer permissive licenses)
        license_name = candidate.get("license", "").lower()
        license_score = 0.0
        if any(l in license_name for l in ["mit", "apache", "bsd", "isc"]):
            license_score = 1.0
        elif any(l in license_name for l in ["lgpl", "mpl"]):
            license_score = 0.5
        elif license_name and license_name != "unknown":
            license_score = 0.2
        
        # Recency scoring (prefer recently updated packages)
        recency_score = 0.0
        if ecosystem == "npm" and candidate.get("date"):
            # Implementation would parse date and score based on recency
            recency_score = 0.5  # Placeholder
        elif ecosystem == "pypi" and candidate.get("releases"):
            # Score based on number of recent releases
            recency_score = min(len(candidate.get("releases", [])) * 0.2, 1.0)
        
        # Weighted composite score
        composite_score = (
            0.25 * quality_score +      # npm quality metric
            0.20 * popularity_score +   # npm popularity
            0.15 * maintenance_score +  # npm maintenance
            0.10 * download_score +     # usage/downloads
            0.20 * security_z_score +   # OpenSSF security score  
            0.05 * license_score +      # license compatibility
            0.05 * recency_score        # recent activity
        )
        
        ranked_candidates.append({
            **candidate,
            "composite_score": composite_score,
            "quality_z": quality_score,
            "popularity_z": popularity_score,
            "maintenance_z": maintenance_score,
            "download_z": download_score,
            "security_z": security_z_score,
            "license_score": license_score,
            "recency_score": recency_score,
        })
    
    # Sort by composite score
    ranked_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Group by capability for better selection
    capability_groups = {}
    for candidate in ranked_candidates:
        capability = candidate.get("capability", "unknown")
        if capability not in capability_groups:
            capability_groups[capability] = []
        capability_groups[capability].append(candidate)
    
    # Select top candidates per capability (balanced selection)
    final_selection = []
    for capability, candidates in capability_groups.items():
        # Take top 3-4 per capability, ensuring ecosystem diversity
        npm_candidates = [c for c in candidates if c.get("ecosystem") == "npm"][:2]
        pypi_candidates = [c for c in candidates if c.get("ecosystem") == "pypi"][:2] 
        final_selection.extend(npm_candidates + pypi_candidates)
    
    return {
        "ranked_packages": final_selection[:20],  # Top 20 overall
        "capability_groups": capability_groups,
        "ranking_summary": {
            "total_evaluated": len(all_candidates),
            "final_selection": len(final_selection),
            "npm_count": len([c for c in final_selection if c.get("ecosystem") == "npm"]),
            "pypi_count": len([c for c in final_selection if c.get("ecosystem") == "pypi"]),
        }
    }
