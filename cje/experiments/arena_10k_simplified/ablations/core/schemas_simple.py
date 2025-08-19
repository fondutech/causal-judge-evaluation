"""Simplified schemas for ablation experiments."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass(frozen=True)
class ExperimentSpec:
    """Minimal experiment specification for ablations."""
    
    # Core required fields
    ablation: str                           # Which ablation (e.g., "oracle_coverage")
    dataset_path: str                       # Path to dataset
    estimator: str                          # Which estimator to use
    
    # Key experimental dimensions
    oracle_coverage: Optional[float] = None # Fraction with oracle labels
    sample_size: Optional[int] = None       # Number of samples
    
    # Reproducibility
    seed: int = 42                          # Random seed
    
    # Optional extra params
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def uid(self) -> str:
        """Generate unique ID for caching."""
        key_dict = {
            "ablation": self.ablation,
            "dataset_path": self.dataset_path,
            "estimator": self.estimator,
            "oracle_coverage": self.oracle_coverage,
            "sample_size": self.sample_size,
            "seed": self.seed,
        }
        key = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across seeds with simplified logic."""
    
    if not results:
        return {}
    
    # Group estimates by policy
    by_policy = {}
    for r in results:
        if "estimates" in r:
            for policy, est in r["estimates"].items():
                if policy not in by_policy:
                    by_policy[policy] = {"estimates": [], "ses": []}
                by_policy[policy]["estimates"].append(est)
                if "standard_errors" in r:
                    by_policy[policy]["ses"].append(r["standard_errors"].get(policy, np.nan))
    
    # Compute aggregates
    aggregated = {
        "n_seeds": len(results),
        "success_rate": sum(1 for r in results if r.get("success", False)) / len(results),
        "by_policy": {}
    }
    
    for policy, data in by_policy.items():
        estimates = np.array(data["estimates"])
        ses = np.array(data["ses"])
        
        aggregated["by_policy"][policy] = {
            "mean_estimate": float(np.mean(estimates)),
            "std_estimate": float(np.std(estimates)),
            "mean_se": float(np.nanmean(ses)),
            "n_valid": len(estimates)
        }
    
    return aggregated