#!/usr/bin/env python3
"""
Run CJE analysis on prepared Arena data

This shows how to use the decoupled loading and calibration approach:
1. Load dataset (rewards optional)
2. Calibrate judge scores OR use oracle labels directly
3. Run CJE estimation with cross-fitting
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import numpy as np
import os
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging based on environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from cje_simplified import (
    load_dataset_from_jsonl,
    PrecomputedSampler,
    CalibratedIPS,
    RawIPS,
    diagnose_weights,
    create_weight_summary_table,
    analyze_extreme_weights,
)

# Import visualization if available
try:
    from cje_simplified import (
        plot_weight_calibration_analysis,
        plot_weight_diagnostics_summary,
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
        "--oracle-coverage",
        type=float,
        default=1.0,
        help="Fraction of oracle labels to use for calibration (0.0-1.0). Default: 1.0",
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
        help="Directory to save visualization plots (defaults to same dir as data file)",
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
        help='JSON config for estimator (e.g., \'{"k_folds": 10, "clip_weight": 50}\')',
    )
    parser.add_argument(
        "--extreme-threshold-high",
        type=float,
        default=100.0,
        help="Weights above this are considered extreme (default: 100.0)",
    )
    parser.add_argument(
        "--extreme-threshold-low",
        type=float,
        default=0.01,
        help="Weights below this are considered extreme (default: 0.01)",
    )

    args = parser.parse_args()

    # Update logging level if debug flag is set
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        # Also set debug for key modules
        logging.getLogger("cje_simplified.calibration.isotonic").setLevel(logging.DEBUG)
        logging.getLogger("cje_simplified.core.calibrated_ips").setLevel(logging.DEBUG)

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
            # Decide between direct oracle use or calibration based on coverage
            if args.oracle_coverage == 1.0:
                print("\n2. Using oracle labels directly as rewards (100% coverage)...")
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
                print(
                    f"\n2. Calibrating judge scores using {args.oracle_coverage:.0%} oracle coverage..."
                )
                # Judge calibration workflow with partial oracle coverage
                try:
                    # Need to implement partial oracle calibration
                    import random
                    from sklearn.isotonic import IsotonicRegression

                    # Set seed for reproducibility
                    random.seed(42)
                    np.random.seed(42)

                    # Extract samples with both judge and oracle
                    samples_with_both = []
                    for i, sample in enumerate(dataset.samples):
                        if (
                            args.judge_field in sample.metadata
                            and args.oracle_field in sample.metadata
                        ):
                            samples_with_both.append(i)

                    if not samples_with_both:
                        raise ValueError(
                            "No samples have both judge scores and oracle labels"
                        )

                    # Select subset for calibration
                    n_oracle = max(
                        2, int(len(samples_with_both) * args.oracle_coverage)
                    )
                    calibration_indices = sorted(
                        random.sample(
                            samples_with_both, min(n_oracle, len(samples_with_both))
                        )
                    )

                    # Extract arrays for calibration
                    judge_scores = []
                    oracle_labels = []
                    for idx in calibration_indices:
                        sample = dataset.samples[idx]
                        judge_scores.append(sample.metadata[args.judge_field])
                        oracle_labels.append(sample.metadata[args.oracle_field])

                    # Fit isotonic regression
                    iso_reg = IsotonicRegression(out_of_bounds="clip")
                    iso_reg.fit(judge_scores, oracle_labels)

                    # Apply calibration to all samples with judge scores
                    calibrated_count = 0
                    for sample in dataset.samples:
                        if args.judge_field in sample.metadata:
                            judge_score = sample.metadata[args.judge_field]
                            sample.reward = float(iso_reg.predict([judge_score])[0])
                            calibrated_count += 1

                    calibrated_dataset = dataset

                    # Report calibration quality
                    all_judge = []
                    all_oracle = []
                    for sample in dataset.samples:
                        if (
                            args.judge_field in sample.metadata
                            and args.oracle_field in sample.metadata
                            and sample.reward is not None
                        ):
                            all_judge.append(sample.reward)
                            all_oracle.append(sample.metadata[args.oracle_field])

                    if all_judge:
                        rmse = np.sqrt(
                            np.mean(
                                [(j - o) ** 2 for j, o in zip(all_judge, all_oracle)]
                            )
                        )
                        coverage = sum(
                            1
                            for j, o in zip(all_judge, all_oracle)
                            if abs(j - o) <= 0.1
                        ) / len(all_judge)

                        print(
                            f"   ✓ Calibrated {calibrated_count} samples using {len(calibration_indices)} oracle labels"
                        )
                        print(f"   ✓ Calibration RMSE: {rmse:.3f}")
                        print(f"   ✓ Coverage (±0.1): {coverage:.1%}")
                    else:
                        print(
                            f"   ✓ Calibrated {calibrated_count} samples using {len(calibration_indices)} oracle labels"
                        )

                except Exception as e:
                    print(f"\n❌ Calibration failed: {e}")
                    raise ValueError(f"Calibration failed: {e}")

    # Step 3: Run CJE estimation (requires rewards)
    print(f"\n3. Running CJE estimation with {args.estimator}...")

    # Initialize variables that will be used in JSON output
    all_weight_diagnostics: Dict[str, Any] = {}
    best_policy = "base"
    mean_w = 1.0
    max_w = 1.0
    ess = float(dataset.n_samples)
    ess_frac = 1.0

    try:
        sampler = PrecomputedSampler(calibrated_dataset)

        # Initialize the selected estimator
        estimator_config = args.estimator_config or {}

        estimator: Union[CalibratedIPS, RawIPS]
        if args.estimator == "calibrated-ips":
            # Updated API: no more k_folds, uses optimized single-pass
            estimator = CalibratedIPS(sampler)
        elif args.estimator == "raw-ips":
            # Use clip_weight from config or default
            clip_weight = estimator_config.get("clip_weight", 100.0)
            estimator = RawIPS(sampler, clip_weight=clip_weight)
        else:
            raise ValueError(f"Unknown estimator: {args.estimator}")

        results = estimator.fit_and_estimate()

        # Get target policies list for indexing
        target_policies: List[str] = list(sampler.target_policies)

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
        for policy, estimate, se, ci_l, ci_u in zip(
            target_policies,
            results.estimates,
            results.standard_errors,
            ci_lower,
            ci_upper,
        ):
            print(f"   {policy}:")
            print(f"     Estimate: {estimate:.3f}")
            print(f"     Std Error: {se:.3f}")
            print(f"     95% CI: [{ci_l:.3f}, {ci_u:.3f}]")

        # Best policy (including base)
        all_estimates = [base_mean] + list(results.estimates)
        all_policies = ["base"] + target_policies
        best_idx = np.argmax(all_estimates)
        best_policy = all_policies[best_idx]
        print(f"\n   🏆 Best policy: {best_policy}")

        # Compare to oracle ground truth if available
        if args.oracle_field in dataset.samples[0].metadata:
            print(f"\n   📊 Oracle Ground Truth Comparison:")

            # Try to load oracle labels for each policy from response files
            oracle_means = {}
            responses_dir = Path(args.data).parent / "responses"

            # Compute oracle mean for base policy from dataset
            base_oracle_labels = [
                s.metadata[args.oracle_field]
                for s in dataset.samples
                if args.oracle_field in s.metadata
            ]
            if base_oracle_labels:
                oracle_means["base"] = sum(base_oracle_labels) / len(base_oracle_labels)

            # Try to load oracle labels for target policies from their response files
            for policy in target_policies:
                response_file = responses_dir / f"{policy}_responses.jsonl"
                if response_file.exists():
                    try:
                        oracle_labels = []
                        with open(response_file, "r") as f:
                            for line in f:
                                data = json.loads(line)
                                if (
                                    "metadata" in data
                                    and args.oracle_field in data["metadata"]
                                ):
                                    oracle_labels.append(
                                        data["metadata"][args.oracle_field]
                                    )
                        if oracle_labels:
                            oracle_means[policy] = sum(oracle_labels) / len(
                                oracle_labels
                            )
                    except Exception:
                        pass  # Silently skip if can't load

            # If we couldn't load target policy oracle labels, note it
            if len(oracle_means) == 1:
                # Only have base policy oracle labels
                print(f"   Base Policy (Observed):")
                print(f"     CJE Estimate: {base_mean:.3f}")
                print(f"     Oracle Mean:  {oracle_means['base']:.3f}")
                print(f"     Error:        {base_mean - oracle_means['base']:+.3f}")
                print(
                    f"\n   Note: Oracle labels for target policies not available in response files."
                )
                print(f"   CJE uses importance weighting on base policy responses.")
            else:
                # Have oracle labels for multiple policies - show full comparison
                print(
                    f"   {'Policy':<12} {'CJE Estimate':>12} {'Oracle Mean':>12} {'Error':>10}"
                )
                print(f"   {'-'*46}")

                # Show base policy
                oracle_val = oracle_means.get("base", 0.0)
                error = base_mean - oracle_val
                print(
                    f"   {'base':<12} {base_mean:>12.3f} {oracle_val:>12.3f} {error:>+10.3f}"
                )

                # Show target policies
                for policy, estimate in zip(target_policies, results.estimates):
                    if policy in oracle_means:
                        oracle_val = oracle_means[policy]
                        error = estimate - oracle_val
                        print(
                            f"   {policy:<12} {estimate:>12.3f} {oracle_val:>12.3f} {error:>+10.3f}"
                        )
                    else:
                        print(
                            f"   {policy:<12} {estimate:>12.3f} {'N/A':>12} {'N/A':>10}"
                        )

                # Check if CJE identified the correct best policy
                if all(p in oracle_means for p in all_policies):
                    oracle_best = max(oracle_means.items(), key=lambda x: x[1])[0]
                    if oracle_best == best_policy:
                        print(
                            f"\n   ✅ CJE correctly identified {best_policy} as the best policy"
                        )
                    else:
                        print(
                            f"\n   ❌ CJE selected {best_policy}, but oracle shows {oracle_best} is best"
                        )

        # Show weight diagnostics for all policies
        print(f"\n5. Weight diagnostics:")

        # Collect diagnostics for all policies
        all_weight_diagnostics = {}

        # Base policy has uniform weights (no importance sampling)
        base_diag = diagnose_weights(
            np.ones(len(base_rewards)),
            "base",
            expected_weight=1.0,
            extreme_threshold_high=args.extreme_threshold_high,
            extreme_threshold_low=args.extreme_threshold_low,
        )
        all_weight_diagnostics["base"] = base_diag

        # Target policies
        for policy in target_policies:
            weights = estimator.get_weights(policy)
            if weights is not None:
                # Expected weight is 1.0 for clone, None for others
                expected = 1.0 if policy == "clone" else None
                diag = diagnose_weights(
                    weights,
                    policy,
                    expected,
                    extreme_threshold_high=args.extreme_threshold_high,
                    extreme_threshold_low=args.extreme_threshold_low,
                )
                all_weight_diagnostics[policy] = diag

        # Print summary table
        print("\n" + create_weight_summary_table(all_weight_diagnostics))

        # Print detailed diagnostics if any issues found
        has_issues = any(
            d.consistency_flag != "GOOD" for d in all_weight_diagnostics.values()
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

        # Generate extreme weights analysis
        print(f"\n6. Analyzing extreme weights...")

        # Collect raw and calibrated weights for analysis
        analysis_raw_weights = {}
        analysis_cal_weights = {}

        for policy in sampler.target_policies:
            # Get raw weights (works for both CalibratedIPS and RawIPS)
            raw_weights = estimator.get_raw_weights(policy)
            if raw_weights is not None:
                analysis_raw_weights[policy] = raw_weights

            # Get calibrated/final weights
            cal_weights = estimator.get_weights(policy)
            if cal_weights is not None:
                analysis_cal_weights[policy] = cal_weights

        # Generate extreme weights report
        if analysis_raw_weights:
            try:
                # Use same directory as plots for the report
                report_dir = None
                if not args.no_plots:
                    if args.plot_dir:
                        report_dir = Path(args.plot_dir)
                    else:
                        report_dir = Path(args.data).parent / "plots"
                    report_dir.mkdir(parents=True, exist_ok=True)

                json_report, text_report = analyze_extreme_weights(
                    dataset=calibrated_dataset,
                    sampler=sampler,
                    raw_weights_dict=analysis_raw_weights,
                    calibrated_weights_dict=analysis_cal_weights,
                    n_extreme=5,
                    output_dir=report_dir,
                    near_zero_threshold=args.extreme_threshold_low,
                )

                # Print summary of findings
                print(f"   ✓ Analyzed {len(analysis_raw_weights)} policies")
                for policy in analysis_raw_weights.keys():
                    if policy in json_report.get("per_policy_analysis", {}):
                        stats = json_report["per_policy_analysis"][policy]["statistics"]
                        print(
                            f"   ✓ {policy}: {stats['n_clipped_high']} clipped, {stats['n_near_zero']} near-zero"
                        )

                if report_dir:
                    print(
                        f"   ✓ Saved detailed report to {report_dir}/extreme_weights_analysis.txt"
                    )

            except Exception as e:
                print(f"   ⚠️  Could not generate extreme weights analysis: {e}")

        # Generate visualizations by default (unless --no-plots is specified)
        if _viz_available and not args.no_plots:
            # Default plot_dir to same directory as data file
            if args.plot_dir:
                plot_dir = Path(args.plot_dir)
            else:
                # Default to plots/ subdirectory next to the data file
                plot_dir = Path(args.data).parent / "plots"

            print(f"\n7. Generating visualizations in {plot_dir}/...")
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Collect weights for visualization
            weights_dict = {}
            raw_weights_dict = {}
            calibrated_weights_dict = {}

            # Base policy (uniform weights) - only for backward compatibility plots
            weights_dict["base"] = np.ones(len(base_rewards))

            # Target policies only for calibration plots (not base)
            for policy in sampler.target_policies:
                # Get calibrated/final weights
                weights = estimator.get_weights(policy)
                if weights is not None:
                    weights_dict[policy] = weights
                    calibrated_weights_dict[policy] = weights

                # Get raw weights (works for both estimator types)
                raw_weights = estimator.get_raw_weights(policy)
                if raw_weights is not None:
                    raw_weights_dict[policy] = raw_weights
                elif weights is not None:
                    # Fall back to calibrated weights if raw not available
                    raw_weights_dict[policy] = weights

            # Generate plots
            try:
                import matplotlib.pyplot as plt

                if raw_weights_dict and calibrated_weights_dict:
                    # 1. Comprehensive weight calibration analysis (6 panels per policy)
                    fig = plot_weight_calibration_analysis(
                        raw_weights_dict, calibrated_weights_dict
                    )
                    fig.savefig(
                        plot_dir / "weight_calibration_analysis.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    print(
                        f"   ✓ Comprehensive calibration analysis → "
                        f"{plot_dir}/weight_calibration_analysis.png"
                    )
                    plt.close(fig)

                    # 2. Cross-policy diagnostics summary dashboard
                    # Prepare estimates for the summary
                    estimates_dict = {}
                    for i, policy in enumerate(sampler.target_policies):
                        if i < len(results.estimates):
                            estimates_dict[policy] = {
                                "mean": results.estimates[i],
                                "ci_lower": (
                                    ci_lower[i] if "ci_lower" in locals() else 0
                                ),
                                "ci_upper": (
                                    ci_upper[i] if "ci_upper" in locals() else 0
                                ),
                            }

                    fig = plot_weight_diagnostics_summary(
                        raw_weights_dict, calibrated_weights_dict, estimates_dict
                    )
                    fig.savefig(
                        plot_dir / "weight_diagnostics_summary.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    print(
                        f"   ✓ Cross-policy summary dashboard → "
                        f"{plot_dir}/weight_diagnostics_summary.png"
                    )
                    plt.close(fig)

                # Calibration comparison (if calibration was performed)
                if (
                    not args.use_oracle
                    and "cal_result" in locals()
                    and not rewards_exist
                ):
                    # Extract judge scores and oracle labels
                    judge_scores = []
                    oracle_labels = []
                    for s in dataset.samples:
                        if (
                            args.judge_field in s.metadata
                            and args.oracle_field in s.metadata
                        ):
                            judge_scores.append(s.metadata[args.judge_field])
                            oracle_labels.append(s.metadata[args.oracle_field])

                    if judge_scores and oracle_labels:
                        # Get calibrated predictions
                        calibrated_preds = [
                            s.reward
                            for s in calibrated_dataset.samples
                            if s.reward is not None
                        ]

                        if len(calibrated_preds) == len(judge_scores):
                            fig = plot_calibration_comparison(
                                judge_scores=np.array(judge_scores),
                                oracle_labels=np.array(oracle_labels),
                                calibrated_scores=np.array(calibrated_preds),
                            )
                            fig.savefig(
                                plot_dir / "calibration_comparison.png",
                                dpi=150,
                                bbox_inches="tight",
                            )
                            print(
                                f"   ✓ Saved calibration comparison to {plot_dir}/calibration_comparison.png"
                            )

                # Close all figures to free memory
                plt.close("all")

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
    steps_completed = 6  # base steps (including extreme weights)
    if _viz_available and not args.no_plots:
        steps_completed = 7
    print(f"\n✓ Analysis complete! ({steps_completed} steps)")

    # Write results to file if requested
    if args.output:
        results_data: Dict[str, Any] = {
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
                "policies": {},
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
        for policy, estimate, se, ci_l, ci_u in zip(
            target_policies,
            results.estimates,
            results.standard_errors,
            ci_lower,
            ci_upper,
        ):
            results_data["estimation"]["policies"][policy] = {
                "estimate": float(estimate),
                "standard_error": float(se),
                "ci_lower": float(ci_l),
                "ci_upper": float(ci_u),
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

        # Note: Calibration stats are now reported during calibration step
        # Could be enhanced to save these statistics if needed

        # Add visualization info if plots were generated
        if _viz_available and args.plot_dir and not args.no_plots:
            results_data["visualizations"] = {
                "directory": args.plot_dir,
                "plots_generated": True,
            }

        # Write to file
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"\n✓ Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
