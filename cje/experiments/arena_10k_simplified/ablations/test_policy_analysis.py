#!/usr/bin/env python3
"""Test the policy heterogeneity analysis with sample data."""

import json
import numpy as np
from pathlib import Path
from analyze_results import analyze_by_policy, create_policy_heterogeneity_figure


def create_sample_results():
    """Create sample estimator comparison results for testing."""

    # Define methods and policies
    methods = ["IPS", "SNIPS", "Cal-IPS", "DR-CPO", "Cal-DR-CPO", "Stacked-DR"]
    policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]

    # Define characteristic SE patterns for each method x policy
    # This simulates realistic heterogeneity
    se_patterns = {
        "IPS": {
            "clone": 0.180,
            "parallel_universe_prompt": 0.200,
            "premium": 0.190,
            "unhelpful": np.nan,
        },
        "SNIPS": {
            "clone": 0.040,
            "parallel_universe_prompt": 0.050,
            "premium": 0.045,
            "unhelpful": 0.400,
        },
        "Cal-IPS": {
            "clone": 0.020,
            "parallel_universe_prompt": 0.030,
            "premium": 0.025,
            "unhelpful": np.nan,
        },
        "DR-CPO": {
            "clone": 0.015,
            "parallel_universe_prompt": 0.025,
            "premium": 0.020,
            "unhelpful": 0.180,
        },
        "Cal-DR-CPO": {
            "clone": 0.012,
            "parallel_universe_prompt": 0.022,
            "premium": 0.018,
            "unhelpful": 0.150,
        },
        "Stacked-DR": {
            "clone": 0.011,
            "parallel_universe_prompt": 0.020,
            "premium": 0.016,
            "unhelpful": 0.140,
        },
    }

    # ESS patterns (showing overlap quality)
    ess_patterns = {
        "IPS": {
            "clone": 4.5,
            "parallel_universe_prompt": 3.2,
            "premium": 4.0,
            "unhelpful": 0.1,
        },
        "SNIPS": {
            "clone": 4.5,
            "parallel_universe_prompt": 3.2,
            "premium": 4.0,
            "unhelpful": 0.8,
        },
        "Cal-IPS": {
            "clone": 62.7,
            "parallel_universe_prompt": 45.3,
            "premium": 58.2,
            "unhelpful": 0.9,
        },
        "DR-CPO": {
            "clone": 62.7,
            "parallel_universe_prompt": 45.3,
            "premium": 58.2,
            "unhelpful": 0.9,
        },
        "Cal-DR-CPO": {
            "clone": 62.7,
            "parallel_universe_prompt": 45.3,
            "premium": 58.2,
            "unhelpful": 0.9,
        },
        "Stacked-DR": {
            "clone": 62.7,
            "parallel_universe_prompt": 45.3,
            "premium": 58.2,
            "unhelpful": 0.9,
        },
    }

    # Oracle truth values
    oracle_truths = {
        "clone": 0.762,
        "parallel_universe_prompt": 0.771,
        "premium": 0.764,
        "unhelpful": 0.143,
    }

    # Create results
    results = []
    for method in methods:
        # Create estimates with some noise
        estimates = {}
        standard_errors = {}
        ess_absolute = {}
        ess_relative = {}

        for policy in policies:
            se = se_patterns[method][policy]
            if not np.isnan(se):
                # Estimate = oracle + noise based on SE
                estimates[policy] = oracle_truths[policy] + np.random.normal(0, se / 2)
                standard_errors[policy] = se
                n_samples = 1000  # Assume 1000 samples
                ess_absolute[policy] = ess_patterns[method][policy] * n_samples / 100
                ess_relative[policy] = ess_patterns[method][policy]
            else:
                estimates[policy] = np.nan
                standard_errors[policy] = np.nan
                ess_absolute[policy] = 0
                ess_relative[policy] = 0

        result = {
            "spec": {
                "estimator": method.lower().replace("-", "_"),
                "oracle_coverage": 0.1,
                "sample_size": 1000,
                "seed": 42,
            },
            "config": method.lower(),
            "display_name": method,
            "seed": 42,
            "success": True,
            "estimates": estimates,
            "standard_errors": standard_errors,
            "ess_absolute": ess_absolute,
            "ess_relative": ess_relative,
            "oracle_truths": oracle_truths,
            "rmse_vs_oracle": np.mean(
                [
                    abs(estimates[p] - oracle_truths[p])
                    for p in policies
                    if not np.isnan(estimates[p])
                ]
            ),
            "runtime_s": np.random.uniform(10, 30),
        }
        results.append(result)

    return results


def main():
    """Test the policy analysis functions."""

    # Add matplotlib backend to avoid display issues
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend

    print("Creating sample estimator comparison results...")
    results = create_sample_results()

    print(f"Created {len(results)} sample results")
    print()

    # Test the analysis function
    print("Testing analyze_by_policy...")
    df = analyze_by_policy(results)
    print(f"Extracted {len(df)} rows of policy-specific data")
    print("\nFirst few rows:")
    print(df.head(10))
    print()

    # Test the visualization
    print("Testing create_policy_heterogeneity_figure...")
    create_policy_heterogeneity_figure(
        results, save_path=Path("test_heterogeneity.png")
    )

    print("\nTest complete! Check test_heterogeneity.png for the visualization.")
    print()

    # Show key insights
    print("Key Insights from Sample Data:")
    print("-" * 40)
    print("1. Clone policy (minimal shift): Cal-IPS is nearly as good as DR methods")
    print("2. Unhelpful policy (extreme shift): Only DR methods provide estimates")
    print(
        "3. ESS varies dramatically: 60%+ for clone with calibration, <1% for unhelpful"
    )
    print("4. Heterogeneity justifies diagnostic-first approach in the paper")


if __name__ == "__main__":
    main()
