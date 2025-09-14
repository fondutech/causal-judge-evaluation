"""Load and prepare ablation experiment results."""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load experiment results from JSONL file.

    Args:
        results_file: Path to JSONL file containing experiment results

    Returns:
        List of result dictionaries for successful experiments
    """
    results = []
    with open(results_file, "r") as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                if result.get("success", False):
                    results.append(result)
            except json.JSONDecodeError:
                continue
    return results


def add_ablation_config(results: List[Dict[str, Any]]) -> None:
    """Add ablation configuration details to results for easier analysis.

    Extracts:
    - Weight calibration (True/False)
    - IIC usage (True/False)
    - Reward calibration mode (monotone/two-stage/auto)

    Args:
        results: List of result dictionaries (modified in place)
    """
    for result in results:
        spec = result.get("spec", {})
        extra = spec.get("extra", {})

        # Extract configuration
        result["use_weight_calibration"] = extra.get("use_weight_calibration", False)
        result["use_iic"] = extra.get("use_iic", False)
        result["reward_calibration_mode"] = result.get(
            "reward_calibration_used", "unknown"
        )

        # Create a configuration string for grouping
        weight_cal = "WCal" if result["use_weight_calibration"] else "NoWCal"
        iic = "IIC" if result["use_iic"] else "NoIIC"
        reward_mode = result["reward_calibration_mode"]

        result["config_string"] = (
            f"{spec.get('estimator')}_{weight_cal}_{iic}_{reward_mode}"
        )


def add_quadrant_classification(results: List[Dict[str, Any]]) -> None:
    """Add quadrant classification to results based on sample size and oracle coverage.

    Quadrants:
    - Small-LowOracle: ≤1000 samples, ≤25% oracle coverage
    - Small-HighOracle: ≤1000 samples, >25% oracle coverage
    - Large-LowOracle: >1000 samples, ≤25% oracle coverage
    - Large-HighOracle: >1000 samples, >25% oracle coverage

    Args:
        results: List of result dictionaries (modified in place)
    """
    for result in results:
        spec = result.get("spec", {})
        sample_size = spec.get("sample_size", 0)
        oracle_coverage = spec.get("oracle_coverage", 0)

        # Classify into quadrants
        if sample_size <= 1000 and oracle_coverage <= 0.25:
            result["quadrant"] = "Small-LowOracle"
        elif sample_size <= 1000 and oracle_coverage > 0.25:
            result["quadrant"] = "Small-HighOracle"
        elif sample_size > 1000 and oracle_coverage <= 0.25:
            result["quadrant"] = "Large-LowOracle"
        else:  # sample_size > 1000 and oracle_coverage > 0.25
            result["quadrant"] = "Large-HighOracle"


def filter_results(
    results: List[Dict[str, Any]],
    estimator: str = None,
    sample_size: int = None,
    oracle_coverage: float = None,
    quadrant: str = None,
) -> List[Dict[str, Any]]:
    """Filter results by various criteria.

    Args:
        results: List of result dictionaries
        estimator: Filter by estimator name
        sample_size: Filter by exact sample size
        oracle_coverage: Filter by exact oracle coverage
        quadrant: Filter by quadrant classification

    Returns:
        Filtered list of results
    """
    filtered = results

    if estimator is not None:
        filtered = [
            r for r in filtered if r.get("spec", {}).get("estimator") == estimator
        ]

    if sample_size is not None:
        filtered = [
            r for r in filtered if r.get("spec", {}).get("sample_size") == sample_size
        ]

    if oracle_coverage is not None:
        filtered = [
            r
            for r in filtered
            if r.get("spec", {}).get("oracle_coverage") == oracle_coverage
        ]

    if quadrant is not None:
        filtered = [r for r in filtered if r.get("quadrant") == quadrant]

    return filtered
