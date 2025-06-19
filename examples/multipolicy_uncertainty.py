#!/usr/bin/env python3
"""
Multi-Policy Uncertainty-Aware Evaluation Example

This example demonstrates evaluating multiple policies simultaneously
with proper uncertainty quantification and pairwise comparisons.
"""

import numpy as np
from typing import List, Dict, Any

from cje.uncertainty.estimator import UncertaintyAwareDRCPO, UncertaintyEstimatorConfig
from cje.uncertainty.judge import MockUncertaintyJudge


def generate_policy_data(n_samples: int = 200) -> Dict:
    """Generate mock data for multiple policy evaluation."""
    np.random.seed(42)

    # Create uncertainty-aware judge
    judge = MockUncertaintyJudge(
        base_score=0.65,
        base_variance=0.04,
        noise_std=0.1,
    )

    # Generate evaluation samples
    samples = []
    for i in range(n_samples):
        samples.append(
            {
                "context": f"User: Help me understand {['Python', 'JavaScript', 'Rust', 'Go'][i % 4]}",
                "response": f"Assistant: Here's an explanation of the language...",
            }
        )

    # Score with uncertainty
    judge_scores = judge.score_batch(samples, disable_progress=True)

    # Generate mock oracle labels
    oracle_rewards = np.array(
        [np.clip(s.mean + np.random.normal(0, 0.12), 0, 1) for s in judge_scores]
    )

    # Generate importance weights for 4 policies
    # Simulating different distribution shifts
    importance_weights = np.zeros((n_samples, 4))

    # Policy 0: GPT-3.5 (baseline)
    importance_weights[:, 0] = np.exp(np.random.normal(0, 0.8, n_samples))

    # Policy 1: GPT-4 (moderate improvement)
    importance_weights[:, 1] = np.exp(np.random.normal(0.3, 0.9, n_samples))

    # Policy 2: Claude-3 (different distribution)
    importance_weights[:, 2] = np.exp(np.random.normal(-0.2, 1.2, n_samples))

    # Policy 3: Llama-3 (high variance)
    importance_weights[:, 3] = np.exp(np.random.normal(0.1, 1.5, n_samples))

    # Add some extreme weights to test shrinkage
    for policy_idx in range(4):
        high_weight_indices = np.random.choice(n_samples, size=10, replace=False)
        importance_weights[high_weight_indices, policy_idx] *= 5

    return {
        "judge_scores": judge_scores,
        "oracle_rewards": oracle_rewards,
        "importance_weights": importance_weights,
        "policy_names": ["GPT-3.5", "GPT-4", "Claude-3", "Llama-3"],
    }


def main() -> None:
    """Demonstrate multi-policy evaluation with uncertainty."""

    print("=" * 70)
    print("MULTI-POLICY UNCERTAINTY-AWARE EVALUATION")
    print("=" * 70)

    # 1. Generate evaluation data
    print("\n1. Generating evaluation data for 4 policies...")
    data = generate_policy_data(n_samples=300)

    print(f"   - Samples: {len(data['judge_scores'])}")
    print(f"   - Policies: {', '.join(data['policy_names'])}")

    # 2. Configure estimator with uncertainty features
    print("\n2. Configuring uncertainty-aware estimator...")
    config = UncertaintyEstimatorConfig(
        k_folds=5,
        use_variance_shrinkage=True,
        fixed_shrinkage_lambda=0.1,
    )
    estimator = UncertaintyAwareDRCPO(config)

    # 3. Run estimation
    print("\n3. Running uncertainty-aware DR-CPO estimation...")
    result = estimator.fit(
        X=None,
        judge_scores=data["judge_scores"],
        oracle_rewards=data["oracle_rewards"],
        importance_weights=data["importance_weights"],
        policy_names=data["policy_names"],
    )

    # 4. Display overall results
    print("\n4. OVERALL RESULTS")
    print("-" * 70)
    print(result.summary())

    # 5. Pairwise comparisons
    print("\n5. PAIRWISE COMPARISONS")
    print("-" * 70)

    comparisons = [
        ("GPT-4", "GPT-3.5"),
        ("Claude-3", "GPT-4"),
        ("Llama-3", "GPT-3.5"),
    ]

    for policy1, policy2 in comparisons:
        comp = result.pairwise_comparison(policy1, policy2)

        print(f"\n{policy1} vs {policy2}:")
        print(f"   Difference: {comp['difference']:.4f} ± {comp['se_difference']:.4f}")
        print(f"   Z-score: {comp['z_score']:.3f}")
        print(f"   P-value: {comp['p_value']:.4f}")
        print(f"   Significant: {'Yes' if comp['significant_at_0.05'] else 'No'}")
        print(f"   Favors: {comp['favors']}")

    # 6. Individual policy analysis
    print("\n6. INDIVIDUAL POLICY ANALYSIS")
    print("-" * 70)

    for policy_name in data["policy_names"]:
        policy = result.get_policy(policy_name)
        print(f"\n{policy_name}:")

        # Basic metrics
        print(f"   Estimate: {policy.value:.4f}")
        print(f"   Std Error: {policy.se:.4f}")
        print(
            f"   95% CI: [{policy.confidence_interval[0]:.4f}, "
            f"{policy.confidence_interval[1]:.4f}]"
        )

        # Variance decomposition
        decomp = policy.estimate.variance_decomposition
        print(f"   Variance Sources:")
        print(f"      - EIF: {decomp.eif_pct:.1f}%")
        print(f"      - Judge: {decomp.judge_pct:.1f}%")

        # ESS and shrinkage
        print(f"   ESS: {policy.estimate.effective_sample_size:.1f}")
        if policy.estimate.shrinkage_applied:
            print(f"   Shrinkage λ: {policy.estimate.shrinkage_lambda:.3f}")

    # 7. Generate uncertainty report for best policy
    print("\n7. DETAILED UNCERTAINTY REPORT FOR BEST POLICY")
    print("-" * 70)

    best_policy_name = result.rank_policies()[0]
    best_policy = result.get_policy(best_policy_name)
    print(f"\nAnalyzing {best_policy_name} (top-ranked policy)...")

    # Get the index of the best policy for weight extraction
    best_idx = data["policy_names"].index(best_policy_name)

    # Extract data for report
    variances = np.array([s.variance for s in data["judge_scores"]])
    rewards = np.array([s.mean for s in data["judge_scores"]])
    weights = data["importance_weights"][:, best_idx]

    # Create mock comparison (in practice, compare with/without uncertainty)
    se_without_uncertainty = best_policy.se * 0.75

    # Mock report data for demonstration
    report_data: Dict[str, Any] = {
        "concentration": {
            "top_10pct_contribution": 0.45,
            "n_high_variance": 12,
        },
        "warnings": ["Low ESS for some policies"],
        "recommendations": ["Consider adaptive shrinkage"],
    }

    # Display key insights
    print(f"\nVariance Concentration:")
    print(
        f"   Top 10% samples: {report_data['concentration']['top_10pct_contribution']*100:.1f}% of variance"
    )

    if report_data["warnings"]:
        print(f"\nWarnings:")
        warnings_list = list(report_data["warnings"])  # Convert to list for indexing
        for warning in warnings_list[:3]:  # Show top 3
            print(f"   ⚠️  {warning}")

    if report_data["recommendations"]:
        print(f"\nRecommendations:")
        recommendations_list = list(
            report_data["recommendations"]
        )  # Convert to list for indexing
        for rec in recommendations_list[:3]:  # Show top 3
            print(f"   • {rec}")

    # 8. Key takeaways
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)

    takeaways = [
        "✓ Multi-policy evaluation with proper uncertainty quantification",
        "✓ Automatic pairwise comparisons with significance testing",
        "✓ Per-policy variance decomposition and shrinkage",
        "✓ Rich diagnostics identify high-variance samples",
        "✓ Clean API makes complex analysis straightforward",
    ]

    for takeaway in takeaways:
        print(f"\n{takeaway}")

    print("\n✅ Multi-policy evaluation complete!")


if __name__ == "__main__":
    main()
