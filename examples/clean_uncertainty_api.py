#!/usr/bin/env python3
"""
Clean Uncertainty-Aware CJE API Example

This example demonstrates the refactored API where uncertainty is a
first-class citizen throughout the pipeline.
"""

import numpy as np
from typing import List, Dict

from cje.uncertainty.judge import (
    UncertaintyAPIJudge,
    UncertaintyJudgeConfig,
    MockUncertaintyJudge,
)
from cje.uncertainty.estimator import UncertaintyAwareDRCPO, UncertaintyEstimatorConfig


def main() -> None:
    """Demonstrate the clean uncertainty-aware API."""

    print("=" * 60)
    print("CLEAN UNCERTAINTY-AWARE CJE API")
    print("=" * 60)

    # 1. Configure uncertainty-aware judge
    print("\n1. Configuring uncertainty-aware judge...")

    # For demonstration, use mock judge
    # In production, use UncertaintyAPIJudge with real config
    judge = MockUncertaintyJudge(
        base_score=0.75,
        base_variance=0.03,
        noise_std=0.08,
    )

    # Example of real API judge configuration:
    """
    judge_config = UncertaintyJudgeConfig(
        name="uncertainty_gpt4",
        provider="openai",
        model_name="gpt-4",
        template="comprehensive_judge",
        structured_output_schema="UncertainJudgeScore",  # Always uses uncertainty
        beta_concentration=15.0,  # Higher = more confident
        use_adaptive_concentration=True,
        include_uncertainty_prompt=True,
    )
    judge = UncertaintyAPIJudge(judge_config)
    """

    # 2. Generate evaluation data
    print("\n2. Generating evaluation samples...")
    n_samples = 100

    samples = []
    for i in range(n_samples):
        samples.append(
            {
                "context": f"Question: What is the capital of country {i}?",
                "response": f"The capital is City_{i}.",
            }
        )

    # 3. Score with uncertainty
    print("\n3. Scoring samples with uncertainty...")
    judge_scores = judge.score_batch(samples, disable_progress=True)

    # Show sample scores
    print("\nSample judge scores (first 5):")
    for i, score in enumerate(judge_scores[:5]):
        print(f"  Sample {i}: mean={score.mean:.3f}, variance={score.variance:.4f}")

    # 4. Generate mock oracle labels (in practice, from human evaluation)
    print("\n4. Generating oracle labels for calibration...")
    oracle_rewards = np.array(
        [np.clip(s.mean + np.random.normal(0, 0.1), 0, 1) for s in judge_scores]
    )

    # 5. Generate importance weights (π'/π₀)
    print("\n5. Computing importance weights...")
    # Simulate some distribution shift
    log_weights = np.random.normal(0, 1.2, n_samples)
    log_weights[::10] += 2  # Some high-weight samples
    importance_weights = np.exp(log_weights)

    # 6. Configure uncertainty-aware estimator
    print("\n6. Configuring uncertainty-aware estimator...")
    config = UncertaintyEstimatorConfig(
        k_folds=5,
        use_variance_shrinkage=True,
        fixed_shrinkage_lambda=0.1,
    )
    estimator = UncertaintyAwareDRCPO(config)

    # 7. Fit estimator
    print("\n7. Fitting uncertainty-aware DR-CPO estimator...")
    result = estimator.fit(
        X=None,  # Features not used in basic DR-CPO
        judge_scores=judge_scores,
        oracle_rewards=oracle_rewards,
        importance_weights=importance_weights,
        policy_names=["GPT-4 vs GPT-3.5"],
    )

    # 8. Display results
    print("\n8. RESULTS")
    print("-" * 40)

    # Get the single policy result
    policy_result = result.get_policy("GPT-4 vs GPT-3.5")
    if not policy_result:
        raise ValueError("Policy not found")

    estimate = policy_result.value
    se = policy_result.se
    ci_lower = policy_result.confidence_interval[0]
    ci_upper = policy_result.confidence_interval[1]

    print(f"Point estimate: {estimate:.4f}")
    print(f"Standard error: {se:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 9. Generate detailed uncertainty report
    print("\n9. Generating uncertainty diagnostics...")

    # Extract data for report
    variances = np.array([s.variance for s in judge_scores])
    rewards = np.array([s.mean for s in judge_scores])

    # Compare with non-uncertainty SE (mock)
    se_without_uncertainty = se * 0.7  # Typically smaller

    # Mock report data for demonstration
    report_data = {
        "variance_decomposition": {
            "judge_uncertainty": 0.35,
            "sampling_variance": 0.65,
        },
        "diagnostics": {
            "ess_percentage": 42.5,
            "high_weight_samples": 8,
            "gamma_calibration": 1.8,
        },
    }

    # 10. Display report insights
    print("\n10. UNCERTAINTY ANALYSIS")
    print("-" * 40)

    print(f"\nVariance Decomposition:")
    decomp = report_data["variance_decomposition"]
    print(f"  - Judge uncertainty: {decomp['judge_uncertainty']*100:.1f}%")
    print(f"  - Sampling variance: {decomp['sampling_variance']*100:.1f}%")

    print(f"\nImpact of Uncertainty:")
    print(f"  - SE with uncertainty: {se:.4f}")
    print(f"  - SE without uncertainty: {se_without_uncertainty:.4f}")
    print(f"  - SE increase: {(se/se_without_uncertainty - 1)*100:.1f}%")

    print(f"\nDiagnostics:")
    diag = report_data["diagnostics"]
    print(f"  - ESS percentage: {diag['ess_percentage']:.1f}%")
    print(f"  - High weight samples: {diag['high_weight_samples']}")
    print(f"  - Gamma calibration: {diag['gamma_calibration']:.2f}")

    # Example warnings
    warnings = []
    if diag["ess_percentage"] < 50:
        warnings.append(
            f"Low ESS ({diag['ess_percentage']:.1f}%) - estimates may be unstable"
        )
    if diag["gamma_calibration"] > 2:
        warnings.append("High gamma indicates judge may be underconfident")

    if warnings:
        print(f"\nWarnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")

    # Example recommendations
    recommendations = []
    if diag["ess_percentage"] < 50:
        recommendations.append(
            "Consider increasing sample size or reducing policy divergence"
        )
    if len(warnings) > 0:
        recommendations.append("Review judge calibration on validation set")

    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")

    # 11. Key advantages of the clean API
    print("\n" + "=" * 60)
    print("KEY ADVANTAGES OF THE CLEAN API")
    print("=" * 60)

    advantages = [
        "1. Uncertainty is mandatory, not optional - no conditional logic",
        "2. Type-safe data flow with validated schemas",
        "3. Clear separation of concerns (judge, calibration, estimation)",
        "4. Rich diagnostics built into the pipeline",
        "5. Automatic variance calibration and shrinkage",
        "6. No backward compatibility complexity",
    ]

    for advantage in advantages:
        print(f"\n✓ {advantage}")

    print("\n✅ Example complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
