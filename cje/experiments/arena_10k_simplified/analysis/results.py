"""Results formatting and display for CJE analysis.

This module handles formatting estimation results, computing statistics,
and displaying them in a clear, readable format.

Following CLAUDE.md: Do one thing well - this module only handles results display.
"""

import numpy as np
from typing import Any, Dict, List, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from oracle_comparison import (
    load_oracle_ground_truth as local_load_oracle_ground_truth,
    compare_estimates_to_oracle,
    format_oracle_comparison_table,
)


def display_results(
    results: Any,
    calibrated_dataset: Any,
    sampler: Any,
    estimator: Any,
    args: Any,
    dataset: Any,
) -> Dict[str, Any]:
    """Display analysis results and return summary data.

    Args:
        results: EstimationResult object
        calibrated_dataset: Dataset with calibrated rewards
        sampler: PrecomputedSampler
        estimator: Fitted estimator
        args: Command-line arguments
        dataset: Original dataset (for oracle comparison)

    Returns:
        Dictionary with summary statistics and best policy
    """
    target_policies = list(sampler.target_policies)
    base_mean, base_se, base_ci_lower, base_ci_upper = compute_base_statistics(
        calibrated_dataset
    )

    print("\n4. Results:")
    print("   " + "-" * 40)

    # Base policy
    print(f"   base (observed):")
    print(f"     Estimate: {base_mean:.3f}")
    print(f"     Std Error: {base_se:.3f}")
    print(f"     95% CI: [{base_ci_lower:.3f}, {base_ci_upper:.3f}]")

    # Target policies
    ci_lower, ci_upper = results.confidence_interval(alpha=0.05)
    for policy, estimate, se, ci_l, ci_u in zip(
        target_policies, results.estimates, results.standard_errors, ci_lower, ci_upper
    ):
        print(f"   {policy}:")
        print(f"     Estimate: {estimate:.3f}")
        print(f"     Std Error: {se:.3f}")
        print(f"     95% CI: [{ci_l:.3f}, {ci_u:.3f}]")

    # Best policy (handle NaN values properly)
    all_estimates = [base_mean] + list(results.estimates)
    all_policies = ["base"] + target_policies

    # Filter out NaN values for best policy selection
    valid_estimates = [
        (est, pol) for est, pol in zip(all_estimates, all_policies) if not np.isnan(est)
    ]

    if valid_estimates:
        best_estimate, best_policy = max(valid_estimates, key=lambda x: x[0])
        print(f"\n   🏆 Best policy: {best_policy}")
    else:
        print(f"\n   ⚠️ No valid estimates available for best policy selection")
        best_policy = None

    # Add sanity check for extreme estimates
    _check_extreme_estimates(all_policies, all_estimates, base_mean, estimator)

    # Oracle comparison if available
    if args.oracle_field in dataset.samples[0].metadata:
        _display_oracle_comparison(args, dataset, target_policies, base_mean, results)

    return {
        "best_policy": best_policy,
        "base_mean": base_mean,
        "base_se": base_se,
        "base_ci_lower": base_ci_lower,
        "base_ci_upper": base_ci_upper,
        "target_policies": target_policies,
    }


def compute_base_statistics(
    calibrated_dataset: Any,
) -> Tuple[float, float, float, float]:
    """Compute base policy statistics.

    Args:
        calibrated_dataset: Dataset with rewards

    Returns:
        Tuple of (mean, standard_error, ci_lower, ci_upper)
    """
    base_rewards = [
        s.reward for s in calibrated_dataset.samples if s.reward is not None
    ]
    base_mean = sum(base_rewards) / len(base_rewards) if base_rewards else 0.0
    base_se = (
        np.std(base_rewards, ddof=1) / np.sqrt(len(base_rewards))
        if len(base_rewards) > 1
        else 0.0
    )
    base_ci_lower = base_mean - 1.96 * base_se
    base_ci_upper = base_mean + 1.96 * base_se
    return base_mean, base_se, base_ci_lower, base_ci_upper


def _check_extreme_estimates(
    all_policies: List[str],
    all_estimates: List[float],
    base_mean: float,
    estimator: Any,
) -> None:
    """Check for extreme estimates that may indicate problems.

    Args:
        all_policies: List of all policy names
        all_estimates: List of all estimates
        base_mean: Base policy mean
        estimator: Fitted estimator (for getting weights)
    """
    EXTREME_DIFF_THRESHOLD = 0.3  # >30% difference is suspicious
    NEAR_ZERO_WEIGHT_THRESHOLD = 1e-10
    EXTREME_CONCENTRATION_THRESHOLD = 0.9

    for policy, estimate in zip(all_policies, all_estimates):
        if (
            not np.isnan(estimate)
            and abs(estimate - base_mean) > EXTREME_DIFF_THRESHOLD
        ):
            print(
                f"\n   ⚠️ WARNING: {policy} estimate ({estimate:.3f}) differs greatly from base ({base_mean:.3f})"
            )
            print(
                f"      This may indicate estimation failure or extreme distribution shift"
            )
            if policy != "base":
                # Check weight concentration for this policy
                weights = estimator.get_weights(policy)
                if weights is not None:
                    near_zero = np.sum(weights < NEAR_ZERO_WEIGHT_THRESHOLD) / len(
                        weights
                    )
                    if near_zero > EXTREME_CONCENTRATION_THRESHOLD:
                        print(
                            f"      Likely cause: {near_zero:.1%} of samples have near-zero weight"
                        )


def _display_oracle_comparison(
    args: Any,
    dataset: Any,
    target_policies: List[str],
    base_mean: float,
    results: Any,
) -> None:
    """Display comparison with oracle ground truth if available.

    Args:
        args: Command-line arguments
        dataset: Original dataset with oracle labels
        target_policies: List of target policies
        base_mean: Base policy mean
        results: EstimationResult object
    """
    print(f"\n   📊 Oracle Ground Truth Comparison:")
    oracle_means = load_oracle_ground_truth(args, dataset, target_policies)

    if oracle_means:
        # Build estimates dictionary including base
        all_estimates_dict = {"base": base_mean}
        for i, policy in enumerate(target_policies):
            all_estimates_dict[policy] = results.estimates[i]

        # Use core library comparison function
        comparison = compare_estimates_to_oracle(all_estimates_dict, oracle_means)

        # Format and display using core library function
        formatted_table = format_oracle_comparison_table(comparison, precision=3)
        for line in formatted_table.split("\n"):
            print(f"   {line}")


def load_oracle_ground_truth(
    args: Any, dataset: Any, target_policies: List[str]
) -> Dict[str, float]:
    """Load oracle ground truth values for comparison.

    Args:
        args: Command-line arguments (contains data path and oracle field)
        dataset: Dataset object
        target_policies: List of target policies

    Returns:
        Dictionary mapping policy names to oracle mean values
    """
    # Use local function (experiment-specific)
    result: Dict[str, float] = local_load_oracle_ground_truth(
        args.data,
        dataset,
        target_policies,
        args.oracle_field,
        responses_dir=str(Path(args.data).parent / "responses"),
    )
    return result
