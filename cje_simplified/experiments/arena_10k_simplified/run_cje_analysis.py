#!/usr/bin/env python3
"""
Run CJE analysis on prepared Arena data using the new architecture.

This shows how to use the decoupled loading and calibration approach:
1. Load dataset (rewards optional)
2. Calibrate judge scores OR use oracle labels directly
3. Run CJE estimation with cross-fitting
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje_simplified import (
    load_dataset_from_jsonl,
    calibrate_dataset,
    PrecomputedSampler,
    CalibratedIPS,
)


def main() -> int:
    """Run complete CJE analysis workflow."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CJE analysis on Arena data")
    parser.add_argument(
        "--data",
        default="data/cje_dataset.jsonl",
        help="Path to prepared CJE dataset",
    )
    parser.add_argument(
        "--use-oracle",
        action="store_true",
        help="Use oracle labels directly as rewards (skip calibration)",
    )
    parser.add_argument(
        "--judge-field",
        default="judge_score",
        help="Field containing judge scores",
    )
    parser.add_argument(
        "--oracle-field",
        default="oracle_label",
        help="Field containing oracle labels",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-fitting folds",
    )

    args = parser.parse_args()

    print("Running CJE Analysis")
    print("=" * 50)

    # Step 1: Load data (no rewards required)
    print("\n1. Loading dataset...")
    dataset = load_dataset_from_jsonl(args.data)
    print(f"   ✓ Loaded {dataset.n_samples} samples")
    print(f"   ✓ Target policies: {dataset.target_policies}")

    # Step 2: Handle rewards based on workflow
    if args.use_oracle:
        print("\n2. Using oracle labels directly as rewards...")
        # Direct oracle workflow - assign oracle labels as rewards
        oracle_count = 0
        for sample in dataset.samples:
            if args.oracle_field in sample.metadata:
                sample.reward = float(sample.metadata[args.oracle_field])
                oracle_count += 1

        if oracle_count == 0:
            raise ValueError(f"No oracle labels found in field '{args.oracle_field}'")

        print(f"   ✓ Assigned {oracle_count} oracle labels as rewards")
        calibrated_dataset = dataset

    else:
        print("\n2. Calibrating judge scores to oracle labels...")
        # Judge calibration workflow
        try:
            calibrated_dataset, cal_result = calibrate_dataset(
                dataset,
                judge_field=args.judge_field,
                oracle_field=args.oracle_field,
                k_folds=args.n_folds,
            )
            print(f"   ✓ Calibrated using {cal_result.n_oracle} oracle samples")
            print(f"   ✓ Calibration RMSE: {cal_result.calibration_rmse:.3f}")
            print(f"   ✓ Coverage (±0.1): {cal_result.coverage_at_01:.1%}")

        except ValueError as e:
            print(f"\n❌ Calibration failed: {e}")
            print("\nTrying pre-calibrated rewards workflow...")

            # Check if rewards already exist
            rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)
            if rewards_exist > 0:
                print(f"   ✓ Found {rewards_exist} pre-calibrated rewards")
                calibrated_dataset = dataset
            else:
                raise ValueError("No rewards found and calibration failed")

    # Step 3: Run CJE estimation (requires rewards)
    print("\n3. Running CJE estimation...")
    try:
        sampler = PrecomputedSampler(calibrated_dataset)
        estimator = CalibratedIPS(sampler, k_folds=args.n_folds)
        results = estimator.fit_and_estimate()

        # Display results
        print("\n4. Results:")
        print("   " + "-" * 40)

        # Display results for each policy
        policy_results = results.policy_results
        for policy, result in policy_results.items():
            print(f"   {policy}:")
            print(f"     Estimate: {result.point_estimate:.3f}")
            print(f"     Std Error: {result.standard_error:.3f}")
            print(f"     95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
            print(f"     Relative efficiency: {result.relative_efficiency:.1%}")

        # Best policy
        best_policy = max(policy_results.items(), key=lambda x: x[1].point_estimate)[0]
        print(f"\n   🏆 Best policy: {best_policy}")

        # Show weight diagnostics for best policy
        print(f"\n5. Weight diagnostics for best policy ({best_policy}):")
        best_result = policy_results[best_policy]
        if hasattr(best_result, "weights") and best_result.weights is not None:
            weights = best_result.weights
            mean_w = weights.mean()
            max_w = weights.max()
            ess = best_result.effective_sample_size
            ess_frac = ess / len(weights)

            print(f"     Mean weight: {mean_w:.3f}")
            print(f"     Max weight: {max_w:.3f}")
            print(f"     Effective sample size: {ess:.0f} ({ess_frac:.1%})")

    except ValueError as e:
        print(f"\n❌ Estimation failed: {e}")
        print("\nThis usually means the dataset is missing rewards.")
        print("Please ensure you either:")
        print("  1. Have oracle labels to use directly (--use-oracle)")
        print("  2. Have judge scores and oracle labels for calibration")
        print("  3. Have pre-calibrated rewards in the dataset")
        return 1

    print("\n✓ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
