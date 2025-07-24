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
import numpy as np

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
    import json
    from datetime import datetime

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
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write results (optional, defaults to stdout only)",
    )

    args = parser.parse_args()

    print("Running CJE Analysis")
    print("=" * 50)

    # Step 1: Load data (no rewards required)
    print("\n1. Loading dataset...")
    dataset = load_dataset_from_jsonl(args.data)
    print(f"   ✓ Loaded {dataset.n_samples} samples")
    print(f"   ✓ Target policies: {dataset.target_policies}")

    # Step 2: Handle rewards
    # Check if rewards already exist in the dataset
    rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)

    if rewards_exist > 0:
        print(f"\n2. Using pre-computed rewards from dataset...")
        print(f"   ✓ Found {rewards_exist}/{dataset.n_samples} samples with rewards")
        calibrated_dataset = dataset

        # Report reward statistics
        rewards = [s.reward for s in dataset.samples if s.reward is not None]
        print(f"   ✓ Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        print(f"   ✓ Mean reward: {sum(rewards)/len(rewards):.3f}")

        # If we have oracle labels, report calibration quality
        if all(args.oracle_field in s.metadata for s in dataset.samples):
            oracle_labels = [
                s.metadata[args.oracle_field]
                for s in dataset.samples
                if s.reward is not None
            ]
            rmse = np.sqrt(
                np.mean([(r - o) ** 2 for r, o in zip(rewards, oracle_labels)])
            )
            abs_errors = [abs(r - o) for r, o in zip(rewards, oracle_labels)]
            coverage_01 = sum(1 for e in abs_errors if e <= 0.1) / len(abs_errors)

            print(f"\n   Reward quality (vs oracle labels):")
            print(f"   ✓ RMSE: {rmse:.3f}")
            print(f"   ✓ Coverage (±0.1): {coverage_01:.1%}")

    else:
        # Fallback to old behavior for backward compatibility
        if args.use_oracle:
            print("\n2. Using oracle labels directly as rewards...")
            # Direct oracle workflow - assign oracle labels as rewards
            oracle_count = 0
            for sample in dataset.samples:
                if args.oracle_field in sample.metadata:
                    sample.reward = float(sample.metadata[args.oracle_field])
                    oracle_count += 1

            if oracle_count == 0:
                raise ValueError(
                    f"No oracle labels found in field '{args.oracle_field}'"
                )

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

        # Display base policy results first
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

        print(f"   base (observed):")
        print(f"     Estimate: {base_mean:.3f}")
        print(f"     Std Error: {base_se:.3f}")
        print(f"     95% CI: [{base_ci_lower:.3f}, {base_ci_upper:.3f}]")

        # Display results for each target policy
        ci_lower, ci_upper = results.confidence_interval(alpha=0.05)
        for i, policy in enumerate(sampler.target_policies):
            print(f"   {policy}:")
            print(f"     Estimate: {results.estimates[i]:.3f}")
            print(f"     Std Error: {results.standard_errors[i]:.3f}")
            print(f"     95% CI: [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]")

        # Best policy (including base)
        all_estimates = [base_mean] + list(results.estimates)
        all_policies = ["base"] + sampler.target_policies
        best_idx = np.argmax(all_estimates)
        best_policy = all_policies[best_idx]
        print(f"\n   🏆 Best policy: {best_policy}")

        # Show weight diagnostics for best policy
        print(f"\n5. Weight diagnostics for best policy ({best_policy}):")
        if best_policy == "base":
            # Base policy has uniform weights
            mean_w = 1.0
            max_w = 1.0
            ess = float(len(base_rewards))
            ess_frac = 1.0
            print(f"     Mean weight: {mean_w:.3f}")
            print(f"     Max weight: {max_w:.3f}")
            print(f"     Effective sample size: {ess:.0f} ({ess_frac:.1%})")
        else:
            weights = estimator.get_weights(best_policy)
            if weights is not None:
                mean_w = weights.mean()
                max_w = weights.max()
                # Calculate effective sample size
                ess = (weights.sum() ** 2) / (weights**2).sum()
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

    # Write results to file if requested
    if args.output:
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "path": args.data,
                "n_samples": dataset.n_samples,
                "target_policies": dataset.target_policies,
            },
            "workflow": (
                "pre_computed_rewards"
                if rewards_exist > 0
                else ("oracle_direct" if args.use_oracle else "judge_calibration")
            ),
            "estimation": {"n_folds": args.n_folds, "policies": {}},
            "best_policy": best_policy,
            "weight_diagnostics": {
                "mean_weight": float(mean_w),
                "max_weight": float(max_w),
                "effective_sample_size": float(ess),
                "effective_sample_size_fraction": float(ess_frac),
            },
        }

        # Add base policy results
        results_data["estimation"]["policies"]["base"] = {
            "estimate": float(base_mean),
            "standard_error": float(base_se),
            "ci_lower": float(base_ci_lower),
            "ci_upper": float(base_ci_upper),
            "type": "observed",
            "n_samples": len(base_rewards),
        }

        # Add per-policy results
        for i, policy in enumerate(sampler.target_policies):
            results_data["estimation"]["policies"][policy] = {
                "estimate": float(results.estimates[i]),
                "standard_error": float(results.standard_errors[i]),
                "ci_lower": float(ci_lower[i]),
                "ci_upper": float(ci_upper[i]),
                "type": "counterfactual",
            }

        # Add calibration stats if available
        if not args.use_oracle and "cal_result" in locals():
            results_data["calibration"] = {
                "n_oracle": cal_result.n_oracle,
                "calibration_rmse": float(cal_result.calibration_rmse),
                "coverage_at_01": float(cal_result.coverage_at_01),
            }

        # Write to file
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\n✓ Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
