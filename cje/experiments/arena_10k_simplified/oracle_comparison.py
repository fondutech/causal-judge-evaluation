"""Oracle ground truth comparison utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


def load_oracle_ground_truth(
    data_path: str,
    dataset: Any,
    target_policies: List[str],
    oracle_field: str = "oracle_label",
    responses_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Load oracle ground truth values for comparison.

    Args:
        data_path: Path to main dataset file
        dataset: Dataset object with samples
        target_policies: List of target policy names
        oracle_field: Field name containing oracle labels
        responses_dir: Directory containing response files (defaults to data_path/responses)

    Returns:
        Dictionary mapping policy names to oracle mean values
    """
    oracle_means = {}

    # Get responses directory
    if responses_dir is None:
        responses_path = Path(data_path).parent / "responses"
    else:
        responses_path = Path(responses_dir)

    # Compute base policy oracle mean from dataset
    base_oracle_labels = [
        s.metadata[oracle_field]
        for s in dataset.samples
        if oracle_field in s.metadata and s.metadata[oracle_field] is not None
    ]
    if base_oracle_labels:
        oracle_means["base"] = sum(base_oracle_labels) / len(base_oracle_labels)

    # Load oracle labels for target policies from response files
    for policy in target_policies:
        response_file = responses_path / f"{policy}_responses.jsonl"
        if response_file.exists():
            try:
                oracle_labels = []
                with open(response_file, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if "metadata" in data and oracle_field in data["metadata"]:
                            val = data["metadata"][oracle_field]
                            if val is not None:
                                oracle_labels.append(val)
                if oracle_labels:
                    oracle_means[policy] = sum(oracle_labels) / len(oracle_labels)
            except Exception:
                pass  # Silently skip if can't load

    return oracle_means


def compare_estimates_to_oracle(
    estimates: Dict[str, float],
    oracle_values: Dict[str, float],
    return_best: bool = True,
) -> Dict[str, Any]:
    """Compare CJE estimates against oracle ground truth.

    Args:
        estimates: Dictionary of policy -> estimate value
        oracle_values: Dictionary of policy -> oracle mean value
        return_best: Whether to include best policy comparison

    Returns:
        Dictionary with comparison results including errors and best policy info
    """
    comparison: Dict[str, Any] = {
        "policies": {},
        "has_all_oracle": False,
        "rmse": None,
    }

    errors = []
    for policy, estimate in estimates.items():
        if policy in oracle_values:
            oracle_val = oracle_values[policy]
            error = estimate - oracle_val
            comparison["policies"][policy] = {
                "estimate": estimate,
                "oracle": oracle_val,
                "error": error,
                "relative_error": error / oracle_val if oracle_val != 0 else None,
            }
            errors.append(error)
        else:
            comparison["policies"][policy] = {
                "estimate": estimate,
                "oracle": None,
                "error": None,
                "relative_error": None,
            }

    # Compute RMSE if we have errors
    if errors:
        comparison["rmse"] = float(np.sqrt(np.mean(np.array(errors) ** 2)))

    # Check if we have all oracle values
    comparison["has_all_oracle"] = all(
        policy in oracle_values for policy in estimates.keys()
    )

    # Find best policies
    if return_best and estimates and oracle_values:
        best_estimate_policy = max(estimates.items(), key=lambda x: x[1])[0]
        comparison["best_estimate_policy"] = best_estimate_policy

        # Only compute oracle best if we have oracle values for estimated policies
        oracle_for_estimated = {
            p: v for p, v in oracle_values.items() if p in estimates
        }
        if oracle_for_estimated:
            best_oracle_policy = max(oracle_for_estimated.items(), key=lambda x: x[1])[
                0
            ]
            comparison["best_oracle_policy"] = best_oracle_policy
            comparison["identified_correct_best"] = (
                best_estimate_policy == best_oracle_policy
            )

    return comparison


def format_oracle_comparison_table(
    comparison: Dict[str, Any], precision: int = 3
) -> str:
    """Format oracle comparison results as a readable table.

    Args:
        comparison: Results from compare_estimates_to_oracle
        precision: Number of decimal places for display

    Returns:
        Formatted string table
    """
    lines = []

    # Header
    lines.append(
        f"{'Policy':<20} {'CJE Estimate':>12} {'Oracle Mean':>12} {'Error':>10}"
    )
    lines.append("-" * 54)

    # Policy rows
    for policy, data in comparison["policies"].items():
        estimate_str = f"{data['estimate']:.{precision}f}"

        if data["oracle"] is not None:
            oracle_str = f"{data['oracle']:.{precision}f}"
            error_str = f"{data['error']:+.{precision}f}"
        else:
            oracle_str = "N/A"
            error_str = "N/A"

        lines.append(
            f"{policy:<20} {estimate_str:>12} {oracle_str:>12} {error_str:>10}"
        )

    # Summary statistics
    if comparison.get("rmse") is not None:
        lines.append("")
        lines.append(f"RMSE: {comparison['rmse']:.{precision}f}")

    # Best policy comparison
    if "identified_correct_best" in comparison:
        lines.append("")
        if comparison["identified_correct_best"]:
            lines.append(
                f"✅ CJE correctly identified {comparison['best_estimate_policy']} as best"
            )
        else:
            lines.append(
                f"❌ CJE selected {comparison['best_estimate_policy']}, "
                f"but oracle shows {comparison['best_oracle_policy']} is best"
            )

    return "\n".join(lines)
