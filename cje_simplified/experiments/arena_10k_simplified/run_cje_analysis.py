#!/usr/bin/env python3
"""
Run CJE analysis on prepared Arena data using the new architecture.

This shows how to use the decoupled loading and calibration approach.
"""

import sys
from pathlib import Path

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
        "--k-folds",
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
                k_folds=args.k_folds,
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
        estimator = CalibratedIPS(sampler, k_folds=args.k_folds)
        results = estimator.fit_and_estimate()

        # Display results
        print("\n4. Results:")
        print("   " + "-" * 40)
        for i, policy in enumerate(dataset.target_policies):
            estimate = results.estimates[i]
            stderr = results.standard_errors[i]
            ci_lower, ci_upper = estimate - 1.96 * stderr, estimate + 1.96 * stderr

            print(f"   {policy}:")
            print(f"     Estimate: {estimate:.3f}")
            print(f"     Std Error: {stderr:.3f}")
            print(f"     95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Best policy
        best_idx = results.best_policy()
        print(f"\n   🏆 Best policy: {dataset.target_policies[best_idx]}")

        # Get weight diagnostics
        print("\n5. Weight diagnostics:")
        for policy in dataset.target_policies:
            weights = estimator.get_weights(policy)
            if weights is not None:
                mean_w = weights.mean()
                max_w = weights.max()
                ess = (weights.sum()) ** 2 / (weights**2).sum()
                ess_frac = ess / len(weights)

                print(f"   {policy}:")
                print(f"     Mean weight: {mean_w:.3f}")
                print(f"     Max weight: {max_w:.3f}")
                print(f"     ESS fraction: {ess_frac:.1%}")

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
