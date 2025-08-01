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
import os
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging based on environment variable
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from cje_simplified import (
    load_dataset_from_jsonl,
    calibrate_dataset,
    PrecomputedSampler,
    CalibratedIPS,
    RawIPS,
    diagnose_weights,
    create_weight_summary_table,
)

# Import visualization if available
try:
    from cje_simplified import (
        plot_weight_distributions,
        plot_ess_comparison,
        plot_weight_summary,
        plot_calibration_comparison,
    )
    _viz_available = True
except ImportError:
    _viz_available = False


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
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to write results (optional, defaults to stdout only)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        help="Directory to save visualization plots (requires matplotlib)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation even if matplotlib is available",
    )
    parser.add_argument(
        "--estimator",
        choices=["calibrated-ips", "raw-ips"],
        default="calibrated-ips",
        help="Estimation method to use (default: calibrated-ips)",
    )
    parser.add_argument(
        "--estimator-config",
        type=json.loads,
        help="JSON config for estimator (e.g., '{\"k_folds\": 10, \"clip_weight\": 50}')",
    )

    args = parser.parse_args()
    
    # Update logging level if debug flag is set
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also set debug for key modules
        logging.getLogger('cje_simplified.calibration.isotonic').setLevel(logging.DEBUG)
        logging.getLogger('cje_simplified.core.calibrated_ips').setLevel(logging.DEBUG)

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
    print(f"\n3. Running CJE estimation with {args.estimator}...")
    
    # Initialize variables that will be used in JSON output
    all_weight_diagnostics = {}
    best_policy = "base"
    mean_w = 1.0
    max_w = 1.0
    ess = float(dataset.n_samples)
    ess_frac = 1.0
    
    try:
        sampler = PrecomputedSampler(calibrated_dataset)
        
        # Initialize the selected estimator
        estimator_config = args.estimator_config or {}
        
        if args.estimator == "calibrated-ips":
            # Use n_folds from command line if provided
            k_folds = estimator_config.get("k_folds", args.n_folds)
            estimator = CalibratedIPS(sampler, k_folds=k_folds)
        elif args.estimator == "raw-ips":
            # Use clip_weight from config or default
            clip_weight = estimator_config.get("clip_weight", 100.0)
            estimator = RawIPS(sampler, clip_weight=clip_weight)
        else:
            raise ValueError(f"Unknown estimator: {args.estimator}")
            
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

        # Show weight diagnostics for all policies
        print(f"\n5. Weight diagnostics:")
        
        # Collect diagnostics for all policies
        all_weight_diagnostics = {}
        
        # Base policy has uniform weights (no importance sampling)
        base_diag = diagnose_weights(
            np.ones(len(base_rewards)), 
            "base", 
            expected_weight=1.0
        )
        all_weight_diagnostics["base"] = base_diag
        
        # Target policies
        for policy in sampler.target_policies:
            weights = estimator.get_weights(policy)
            if weights is not None:
                # Expected weight is 1.0 for clone, None for others
                expected = 1.0 if policy == "clone" else None
                diag = diagnose_weights(weights, policy, expected)
                all_weight_diagnostics[policy] = diag
        
        # Print summary table
        print("\n" + create_weight_summary_table(all_weight_diagnostics))
        
        # Print detailed diagnostics if any issues found
        has_issues = any(
            d.consistency_flag != "GOOD" 
            for d in all_weight_diagnostics.values()
        )
        if has_issues:
            print("\n   ⚠️  Weight diagnostics warnings:")
            for policy, diag in all_weight_diagnostics.items():
                if diag.consistency_flag != "GOOD":
                    print(f"\n   {diag.summary()}")
        
        # Store diagnostics for the best policy for JSON output
        best_diag = all_weight_diagnostics.get(best_policy)
        if best_diag:
            mean_w = best_diag.mean_weight
            max_w = best_diag.max_weight
            ess = best_diag.ess_fraction * len(base_rewards)
            ess_frac = best_diag.ess_fraction
        else:
            # Fallback values
            mean_w = 1.0
            max_w = 1.0
            ess = float(len(base_rewards))
            ess_frac = 1.0

        # Generate visualizations if requested
        if _viz_available and args.plot_dir and not args.no_plots:
            print("\n6. Generating visualizations...")
            from pathlib import Path
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect weights for visualization
            weights_dict = {}
            
            # Base policy (uniform weights)
            weights_dict["base"] = np.ones(len(base_rewards))
            
            # Target policies
            for policy in sampler.target_policies:
                weights = estimator.get_weights(policy)
                if weights is not None:
                    weights_dict[policy] = weights
            
            # Generate plots
            try:
                # Weight distributions
                fig = plot_weight_distributions(weights_dict)
                fig.savefig(plot_dir / "weight_distributions.png", dpi=150, bbox_inches='tight')
                print(f"   ✓ Saved weight distributions to {plot_dir}/weight_distributions.png")
                
                # ESS comparison
                fig = plot_ess_comparison(weights_dict)
                fig.savefig(plot_dir / "ess_comparison.png", dpi=150, bbox_inches='tight')
                print(f"   ✓ Saved ESS comparison to {plot_dir}/ess_comparison.png")
                
                # Weight summary
                fig = plot_weight_summary(weights_dict)
                fig.savefig(plot_dir / "weight_summary.png", dpi=150, bbox_inches='tight')
                print(f"   ✓ Saved weight summary to {plot_dir}/weight_summary.png")
                
                # Calibration comparison (if calibration was performed)
                if not args.use_oracle and "cal_result" in locals() and not rewards_exist:
                    # Extract judge scores and oracle labels
                    judge_scores = []
                    oracle_labels = []
                    for s in dataset.samples:
                        if args.judge_field in s.metadata and args.oracle_field in s.metadata:
                            judge_scores.append(s.metadata[args.judge_field])
                            oracle_labels.append(s.metadata[args.oracle_field])
                    
                    if judge_scores and oracle_labels:
                        # Get calibrated predictions
                        calibrated_preds = [s.reward for s in calibrated_dataset.samples if s.reward is not None]
                        
                        if len(calibrated_preds) == len(judge_scores):
                            fig = plot_calibration_comparison(
                                judge_scores=np.array(judge_scores),
                                oracle_labels=np.array(oracle_labels),
                                calibrated_scores=np.array(calibrated_preds)
                            )
                            fig.savefig(plot_dir / "calibration_comparison.png", dpi=150, bbox_inches='tight')
                            print(f"   ✓ Saved calibration comparison to {plot_dir}/calibration_comparison.png")
                
                # Close all figures to free memory
                import matplotlib.pyplot as plt
                plt.close('all')
                
            except Exception as e:
                print(f"   ⚠️  Warning: Failed to generate some plots: {e}")

    except ValueError as e:
        print(f"\n❌ Estimation failed: {e}")
        print("\nThis usually means the dataset is missing rewards.")
        print("Please ensure you either:")
        print("  1. Have oracle labels to use directly (--use-oracle)")
        print("  2. Have judge scores and oracle labels for calibration")
        print("  3. Have pre-calibrated rewards in the dataset")
        return 1

    # Final success message
    steps_completed = 5  # base steps
    if _viz_available and args.plot_dir and not args.no_plots:
        steps_completed = 6
    print(f"\n✓ Analysis complete! ({steps_completed} steps)")

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
            "estimation": {
                "estimator": args.estimator,
                "estimator_config": estimator_config,
                "n_folds": args.n_folds,
                "policies": {}
            },
            "best_policy": best_policy,
            "weight_diagnostics": {
                "best_policy": {
                    "mean_weight": float(mean_w),
                    "max_weight": float(max_w),
                    "effective_sample_size": float(ess),
                    "effective_sample_size_fraction": float(ess_frac),
                },
                "all_policies": {},
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
        
        # Add weight diagnostics for all policies
        for policy, diag in all_weight_diagnostics.items():
            results_data["weight_diagnostics"]["all_policies"][policy] = {
                "mean_weight": float(diag.mean_weight),
                "max_weight": float(diag.max_weight),
                "min_weight": float(diag.min_weight),
                "median_weight": float(diag.median_weight),
                "ess_fraction": float(diag.ess_fraction),
                "extreme_weight_count": int(diag.extreme_weight_count),
                "zero_weight_count": int(diag.zero_weight_count),
                "consistency_flag": diag.consistency_flag,
            }

        # Add calibration stats if available
        if not args.use_oracle and "cal_result" in locals():
            results_data["calibration"] = {
                "n_oracle": cal_result.n_oracle,
                "calibration_rmse": float(cal_result.calibration_rmse),
                "coverage_at_01": float(cal_result.coverage_at_01),
            }
        
        # Add visualization info if plots were generated
        if _viz_available and args.plot_dir and not args.no_plots:
            results_data["visualizations"] = {
                "directory": args.plot_dir,
                "plots_generated": True
            }

        # Write to file
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\n✓ Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
