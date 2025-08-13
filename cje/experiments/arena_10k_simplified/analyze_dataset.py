#!/usr/bin/env python3
"""
Arena 10K analysis script - canonical version.

This script demonstrates comprehensive CJE analysis with:
- All estimators (IPS, Calibrated IPS, DR, MRDR, TMLE)
- Weight diagnostics and extreme weight analysis
- Oracle ground truth comparison
- Full visualization suite
- Experimental features (oracle coverage simulation)

Experiment-specific features (kept local):
- migrate_prompt_id_if_needed(): Legacy data format migration
- Oracle coverage simulation: For research experiments

Core library features used:
- All estimators and calibration
- Weight diagnostics
- Oracle comparison utilities
- Visualization functions
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import os
import logging
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

# Set up logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from cje import (
    load_dataset_from_jsonl,
    calibrate_dataset,
    PrecomputedSampler,
    CalibratedIPS,
    RawIPS,
    DRCPOEstimator,
    MRDREstimator,
    TMLEEstimator,
    FreshDrawDataset,
    FreshDrawSample,
    load_fresh_draws_auto,
    diagnose_weights,
    create_weight_summary_table,
    analyze_extreme_weights,
)

# DR diagnostics are now accessed directly from results.diagnostics
from cje.utils.diagnostics.display import (
    format_dr_diagnostic_summary,
)

# Local oracle comparison utilities (experiment-specific)
from oracle_comparison import (
    load_oracle_ground_truth as local_load_oracle_ground_truth,
    compare_estimates_to_oracle,
    format_oracle_comparison_table,
)

# Import visualization if available
try:
    from cje.visualization import (
        plot_weight_dashboard,
        plot_calibration_comparison,
        plot_policy_estimates,
        plot_dr_dashboard,
    )

    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False


def migrate_prompt_id_if_needed(filepath: str) -> None:
    """Migrate prompt_id from metadata to top-level for backward compatibility.

    This is specific to this experiment's legacy data format.
    """
    path = Path(filepath)
    if not path.exists():
        return

    with open(path, "r") as f:
        first_line = f.readline()
        if not first_line:
            return

        try:
            first_record = json.loads(first_line)
            if "prompt_id" in first_record or "prompt_id" not in first_record.get(
                "metadata", {}
            ):
                return  # No migration needed
        except json.JSONDecodeError:
            return

    print(f"   ⚠️  Migrating prompt_id from metadata to top-level")
    lines = []

    with open(path, "r") as f:
        f.seek(0)  # Reset to beginning
        for line in f:
            if not line.strip():
                lines.append(line)
                continue

            try:
                data = json.loads(line)
                if (
                    "prompt_id" not in data
                    and "metadata" in data
                    and "prompt_id" in data["metadata"]
                ):
                    data["prompt_id"] = data["metadata"]["prompt_id"]
                    del data["metadata"]["prompt_id"]
                lines.append(json.dumps(data) + "\n")
            except json.JSONDecodeError:
                lines.append(line)

    with open(path, "w") as f:
        f.writelines(lines)

    print(f"   ✓ Migration complete")


def add_fresh_draws_to_estimator(
    estimator: Any,
    sampler: PrecomputedSampler,
    data_path: str,
    dataset: Any,
    estimator_config: Dict[str, Any],
) -> None:
    """Add fresh draws to a DR estimator for all target policies.

    This helper loads fresh draws from files. It will fail if fresh draws
    are not available - no synthetic fallback.
    """
    data_dir = Path(data_path).parent

    for policy in sampler.target_policies:
        # Load fresh draws - will raise FileNotFoundError if missing
        try:
            fresh_draws = load_fresh_draws_auto(
                data_dir=data_dir,
                policy=policy,
                verbose=False,
            )
            # Print status - we know these are real draws now
            print(f"     ✓ Loaded {len(fresh_draws.samples)} fresh draws for {policy}")
        except FileNotFoundError as e:
            # Enhance the error message with specific guidance
            raise FileNotFoundError(
                f"No fresh draws found for policy '{policy}' in {data_dir}.\n"
                f"DR/MRDR/TMLE require real fresh draws from teacher forcing.\n"
                f"Options:\n"
                f"  1. Generate fresh draws using generate_fresh_draws.py\n"
                f"  2. Use --estimator calibrated-ips or raw-ips (no fresh draws needed)\n"
                f"  3. Ensure response files exist in {data_dir}/responses/\n"
                f"Original error: {e}"
            ) from e
        estimator.add_fresh_draws(policy, fresh_draws)


def setup_estimator(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any] = None,
) -> Union[CalibratedIPS, RawIPS, DRCPOEstimator, MRDREstimator, TMLEEstimator]:
    """Set up the appropriate estimator based on args."""
    estimator_config = args.estimator_config or {}

    if args.estimator == "calibrated-ips":
        return CalibratedIPS(sampler)

    elif args.estimator == "raw-ips":
        clip_weight = estimator_config.get("clip_weight", 100.0)
        return RawIPS(sampler, clip_weight=clip_weight)

    elif args.estimator == "dr-cpo":
        n_folds = estimator_config.get("n_folds", 5)
        if cal_result and cal_result.calibrator:
            dr_estimator = DRCPOEstimator(
                sampler, n_folds=n_folds, calibrator=cal_result.calibrator
            )
            print("   Using CalibratorBackedOutcomeModel (reusing calibration models)")
        else:
            dr_estimator = DRCPOEstimator(sampler, n_folds=n_folds)
            print("   Using IsotonicOutcomeModel (refitting models)")

        # Load fresh draws
        print("   Loading fresh draws for DR estimation...")
        add_fresh_draws_to_estimator(
            dr_estimator, sampler, args.data, calibrated_dataset, estimator_config
        )

        return dr_estimator

    elif args.estimator == "mrdr":
        n_folds = estimator_config.get("n_folds", 5)
        omega_mode = estimator_config.get("omega_mode", "snips")

        # Ensure cross-fitted calibration for MRDR
        if not cal_result or not cal_result.calibrator:
            print(
                "   ⚠️  MRDR works best with cross-fitted calibration. Re-calibrating..."
            )
            calibrated_dataset, cal_result = calibrate_dataset(
                calibrated_dataset,
                judge_field="judge_score",
                oracle_field="oracle_label",
                enable_cross_fit=True,
                n_folds=n_folds,
            )
            sampler = PrecomputedSampler(calibrated_dataset)

        mrdr_estimator = MRDREstimator(sampler, n_folds=n_folds, omega_mode=omega_mode)
        print(f"   Using MRDR with omega_mode='{omega_mode}'")

        # Load fresh draws
        print("   Loading fresh draws for MRDR estimation...")
        add_fresh_draws_to_estimator(
            mrdr_estimator, sampler, args.data, calibrated_dataset, estimator_config
        )

        return mrdr_estimator

    elif args.estimator == "tmle":
        n_folds = estimator_config.get("n_folds", 5)
        link = estimator_config.get("link", "logit")

        # Ensure cross-fitted calibration for TMLE
        if not cal_result or not cal_result.calibrator:
            print(
                "   ⚠️  TMLE works best with cross-fitted calibration. Re-calibrating..."
            )
            calibrated_dataset, cal_result = calibrate_dataset(
                calibrated_dataset,
                judge_field="judge_score",
                oracle_field="oracle_label",
                enable_cross_fit=True,
                n_folds=n_folds,
            )
            sampler = PrecomputedSampler(calibrated_dataset)

        tmle_estimator = TMLEEstimator(sampler, n_folds=n_folds, link=link)
        print(f"   Using TMLE with link='{link}'")

        # Load fresh draws
        print("   Loading fresh draws for TMLE estimation...")
        add_fresh_draws_to_estimator(
            tmle_estimator, sampler, args.data, calibrated_dataset, estimator_config
        )

        return tmle_estimator

    else:
        raise ValueError(f"Unknown estimator: {args.estimator}")


def compute_base_statistics(
    calibrated_dataset: Any,
) -> Tuple[float, float, float, float]:
    """Compute base policy statistics."""
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


def load_oracle_ground_truth(
    args: Any, dataset: Any, target_policies: List[str]
) -> Dict[str, float]:
    """Load oracle ground truth values for comparison."""
    # Use local function (experiment-specific)
    result: Dict[str, float] = local_load_oracle_ground_truth(
        args.data,
        dataset,
        target_policies,
        args.oracle_field,
        responses_dir=str(Path(args.data).parent / "responses"),
    )
    return result


def display_results(
    results: Any,
    calibrated_dataset: Any,
    sampler: PrecomputedSampler,
    estimator: Any,
    args: Any,
    dataset: Any,
) -> Dict[str, Any]:
    """Display analysis results and return summary data."""
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

    # Best policy
    all_estimates = [base_mean] + list(results.estimates)
    all_policies = ["base"] + target_policies
    best_idx = np.argmax(all_estimates)
    best_policy = all_policies[best_idx]
    print(f"\n   🏆 Best policy: {best_policy}")

    # Oracle comparison if available
    if args.oracle_field in dataset.samples[0].metadata:
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

    return {
        "best_policy": best_policy,
        "base_mean": base_mean,
        "base_se": base_se,
        "base_ci_lower": base_ci_lower,
        "base_ci_upper": base_ci_upper,
        "target_policies": target_policies,
    }


def display_weight_diagnostics(
    estimator: Any, sampler: PrecomputedSampler, calibrated_dataset: Any, args: Any
) -> Dict[str, Any]:
    """Display weight diagnostics and return diagnostic data."""
    print(f"\n5. Weight diagnostics:")

    base_rewards = [
        s.reward for s in calibrated_dataset.samples if s.reward is not None
    ]
    all_weight_diagnostics = {}

    # Base policy (uniform weights)
    base_diag = diagnose_weights(
        np.ones(len(base_rewards)),
        "base",
        extreme_quantile=0.99,
    )
    all_weight_diagnostics["base"] = base_diag

    # Target policies
    for policy in sampler.target_policies:
        weights = estimator.get_weights(policy)
        if weights is not None:
            diag = diagnose_weights(
                weights,
                policy,
                extreme_quantile=0.99,
            )
            all_weight_diagnostics[policy] = diag

    # Print summary table
    print("\n" + create_weight_summary_table(all_weight_diagnostics))

    # Print warnings if issues found
    has_issues = any(
        d.consistency_flag != "GOOD" for d in all_weight_diagnostics.values()
    )
    if has_issues:
        print("\n   ⚠️  Weight diagnostics warnings:")
        for policy, diag in all_weight_diagnostics.items():
            if diag.consistency_flag != "GOOD":
                print(f"\n   {diag.summary()}")

    return all_weight_diagnostics


def display_dr_diagnostics(results: Any, args: Any) -> None:
    """Display DR diagnostics if available."""
    # Check if we have DR diagnostics in the new format (DRDiagnostics object)
    if hasattr(results, "diagnostics") and results.diagnostics is not None:
        from cje.data.diagnostics import DRDiagnostics

        if isinstance(results.diagnostics, DRDiagnostics):
            print(f"\n6. Doubly Robust diagnostics:")
            summary = format_dr_diagnostic_summary(results.diagnostics)
            for line in summary.split("\n"):
                print(f"   {line}")

            # Check for issues
            if results.diagnostics.worst_if_tail_ratio > 100:
                print("\n   ⚠️  Warning: Heavy-tailed influence functions detected")
                print(
                    "      Consider using more fresh draws or checking policy overlap"
                )
            return

    # Fallback to legacy format
    if (
        args.estimator in ["dr-cpo", "mrdr", "tmle"]
        and "dr_diagnostics" in results.metadata
    ):
        print(f"\n6. Doubly Robust diagnostics:")
        # Use the dr_diagnostics directly from metadata
        dr_diagnostics = results.metadata["dr_diagnostics"]
        summary = format_dr_diagnostic_summary(dr_diagnostics)

        for line in summary.split("\n"):
            print(f"   {line}")

        # Check for issues
        if isinstance(dr_diagnostics, dict):
            worst_tail = (
                max(
                    d.get("if_tail_ratio_99_5", 0)
                    for d in dr_diagnostics.values()
                    if isinstance(d, dict)
                )
                if dr_diagnostics
                else 0
            )
            if worst_tail > 100:
                print("\n   ⚠️  Warning: Heavy-tailed influence functions detected")
                print(
                    "      Consider using more fresh draws or checking policy overlap"
                )

        if args.estimator == "tmle" and "tmle_max_score_z" in dr_diagnostics:
            if dr_diagnostics["tmle_max_score_z"] > 2:
                print("\n   ⚠️  Warning: TMLE orthogonality not achieved (|z| > 2)")
                print("      Targeting may not have fully converged")


def analyze_extreme_weights_report(
    estimator: Any, sampler: PrecomputedSampler, calibrated_dataset: Any, args: Any
) -> None:
    """Generate extreme weights analysis report."""
    step_num = 7 if args.estimator in ["dr-cpo", "mrdr", "tmle"] else 6
    print(f"\n{step_num}. Analyzing extreme weights...")

    analysis_raw_weights = {}
    analysis_cal_weights = {}

    for policy in sampler.target_policies:
        raw_weights = estimator.get_raw_weights(policy)
        if raw_weights is not None:
            analysis_raw_weights[policy] = raw_weights

        cal_weights = estimator.get_weights(policy)
        if cal_weights is not None:
            analysis_cal_weights[policy] = cal_weights

    if analysis_raw_weights:
        try:
            report_dir = None
            if not args.no_plots:
                report_dir = (
                    Path(args.plot_dir)
                    if args.plot_dir
                    else Path(args.data).parent / "plots"
                )
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

            print(f"   ✓ Analyzed {len(analysis_raw_weights)} policies")
            for policy in analysis_raw_weights.keys():
                if policy in json_report.get("per_policy_analysis", {}):
                    stats = json_report["per_policy_analysis"][policy]["statistics"]
                    print(
                        f"   ✓ {policy}: {stats['n_clipped_high']} very high, {stats['n_near_zero']} near-zero"
                    )

            if report_dir:
                print(
                    f"   ✓ Saved detailed report to {report_dir}/extreme_weights_analysis.txt"
                )

        except Exception as e:
            print(f"   ⚠️  Could not generate extreme weights analysis: {e}")


def generate_visualizations(
    results: Any,
    dataset: Any,
    calibrated_dataset: Any,
    estimator: Any,
    sampler: PrecomputedSampler,
    args: Any,
    summary_data: Dict[str, Any],
    cal_result: Any = None,
) -> None:
    """Generate all visualization plots."""
    if not VIZ_AVAILABLE or args.no_plots:
        return

    import matplotlib.pyplot as plt

    plot_dir = (
        Path(args.plot_dir) if args.plot_dir else Path(args.data).parent / "plots"
    )
    step_num = 8 if args.estimator in ["dr-cpo", "mrdr", "tmle"] else 7
    print(f"\n{step_num}. Generating visualizations in {plot_dir}/...")
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Weight dashboard
        raw_weights_dict = {}
        calibrated_weights_dict = {}

        for policy in sampler.target_policies:
            weights = estimator.get_weights(policy)
            if weights is not None:
                calibrated_weights_dict[policy] = weights

            raw_weights = estimator.get_raw_weights(policy)
            if raw_weights is not None:
                raw_weights_dict[policy] = raw_weights
            elif weights is not None:
                raw_weights_dict[policy] = weights

        if raw_weights_dict and calibrated_weights_dict:
            # Pass diagnostics object if available
            fig, _ = plot_weight_dashboard(
                raw_weights_dict,
                calibrated_weights_dict,
                n_samples=sampler.n_valid_samples,
                save_path=plot_dir / "weight_dashboard",
                diagnostics=results.diagnostics,  # Pass the new diagnostics object
            )
            print(f"   ✓ Weight dashboard → {plot_dir}/weight_dashboard.png")
            plt.close(fig)

        # Calibration comparison
        # Use the ORIGINAL dataset (before masking) to get all judge/oracle pairs
        judge_scores = []
        oracle_labels = []
        for s in dataset.samples:
            if args.judge_field in s.metadata and args.oracle_field in s.metadata:
                j_score = s.metadata.get(args.judge_field)
                o_label = s.metadata.get(args.oracle_field)
                if j_score is not None and o_label is not None:
                    judge_scores.append(j_score)
                    oracle_labels.append(o_label)

        if judge_scores and oracle_labels:
            # Get calibrated rewards if calibration was performed
            calibrated_preds = None
            if cal_result is not None:  # Calibration was performed
                # Get rewards for ALL samples (calibration was done on partial oracle labels)
                calibrated_preds_list = []
                for s_orig, s_cal in zip(dataset.samples, calibrated_dataset.samples):
                    # Match samples from original dataset (with all oracle labels)
                    # to calibrated dataset (with rewards)
                    if (
                        args.judge_field in s_orig.metadata
                        and args.oracle_field in s_orig.metadata
                        and s_cal.reward is not None
                    ):
                        calibrated_preds_list.append(s_cal.reward)

                # Use calibrated scores if we have them for all samples
                if len(calibrated_preds_list) == len(judge_scores):
                    calibrated_preds = calibrated_preds_list

            fig = plot_calibration_comparison(
                judge_scores=np.array(judge_scores),
                oracle_labels=np.array(oracle_labels),
                calibrated_scores=(
                    np.array(calibrated_preds) if calibrated_preds else None
                ),
                save_path=plot_dir / "calibration_comparison",
            )
            print(
                f"   ✓ Calibration comparison → {plot_dir}/calibration_comparison.png"
            )
            plt.close(fig)

        # Policy estimates forest plot
        policy_estimates = {"base": summary_data["base_mean"]}
        policy_ses = {"base": summary_data["base_se"]}

        for policy, estimate, se in zip(
            summary_data["target_policies"], results.estimates, results.standard_errors
        ):
            if not np.isnan(estimate):
                policy_estimates[policy] = estimate
                policy_ses[policy] = se

        # Try to get oracle values
        oracle_values = None
        if args.oracle_field in dataset.samples[0].metadata:
            oracle_values = load_oracle_ground_truth(
                args, dataset, summary_data["target_policies"]
            )

        fig = plot_policy_estimates(
            estimates=policy_estimates,
            standard_errors=policy_ses,
            oracle_values=oracle_values,
            base_policy="base",
            save_path=plot_dir / "policy_estimates",
        )
        print(f"   ✓ Policy estimates → {plot_dir}/policy_estimates.png")
        plt.close(fig)

        # DR dashboard
        if (
            args.estimator in ["dr-cpo", "mrdr", "tmle"]
            and "dr_diagnostics" in results.metadata
        ):
            try:
                fig, _ = plot_dr_dashboard(results)
                fig.savefig(plot_dir / "dr_dashboard.png", dpi=150, bbox_inches="tight")
                print(f"   ✓ DR dashboard → {plot_dir}/dr_dashboard.png")
                plt.close(fig)

            except Exception as e:
                print(f"   ⚠️  Could not generate DR dashboard: {e}")

        plt.close("all")

    except Exception as e:
        print(f"   ⚠️  Failed to generate some plots: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()


def main() -> int:
    """Run CJE analysis with improved architecture."""
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Run CJE analysis on Arena data")
    parser.add_argument(
        "--data", default="data/cje_dataset.jsonl", help="Path to dataset"
    )
    parser.add_argument(
        "--use-oracle", action="store_true", help="Use oracle labels directly"
    )
    parser.add_argument(
        "--oracle-coverage", type=float, default=1.0, help="Oracle coverage (0-1)"
    )
    parser.add_argument(
        "--judge-field", default="judge_score", help="Judge score field"
    )
    parser.add_argument(
        "--oracle-field", default="oracle_label", help="Oracle label field"
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Cross-fitting folds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--plot-dir", type=str, help="Plot directory")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")
    parser.add_argument(
        "--estimator",
        choices=["calibrated-ips", "raw-ips", "dr-cpo", "mrdr", "tmle"],
        default="calibrated-ips",
        help="Estimation method",
    )
    parser.add_argument(
        "--estimator-config", type=json.loads, help="Estimator config JSON"
    )
    parser.add_argument("--extreme-threshold-high", type=float, default=100.0)
    parser.add_argument("--extreme-threshold-low", type=float, default=0.01)

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("cje.calibration.isotonic").setLevel(logging.DEBUG)
        logging.getLogger("cje.core.calibrated_ips").setLevel(logging.DEBUG)

    print("Running CJE Analysis")
    print("=" * 50)

    try:
        # Step 1: Load data
        print("\n1. Loading dataset...")
        migrate_prompt_id_if_needed(args.data)  # Local migration for legacy data
        dataset = load_dataset_from_jsonl(args.data)
        print(f"   ✓ Loaded {dataset.n_samples} samples")
        print(f"   ✓ Target policies: {dataset.target_policies}")

        # Step 2: Handle rewards/calibration
        print("\n2. Handling rewards...")
        calibrated_dataset = None
        cal_result = None

        rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)

        if rewards_exist > 0:
            print(f"   ✓ Using {rewards_exist} pre-computed rewards from dataset")
            calibrated_dataset = dataset
        elif args.use_oracle or args.oracle_coverage == 1.0:
            print("   Using oracle labels directly as rewards...")
            oracle_count = 0
            for sample in dataset.samples:
                if args.oracle_field in sample.metadata:
                    sample.reward = float(sample.metadata[args.oracle_field])
                    oracle_count += 1
            print(f"   ✓ Assigned {oracle_count} oracle labels as rewards")
            calibrated_dataset = dataset
        else:
            print(f"   Calibrating with {args.oracle_coverage:.0%} oracle coverage...")
            random.seed(42)
            np.random.seed(42)

            # Mask some oracle labels if partial coverage
            if args.oracle_coverage < 1.0:
                samples_with_oracle = [
                    i
                    for i, s in enumerate(dataset.samples)
                    if args.oracle_field in s.metadata
                    and s.metadata[args.oracle_field] is not None
                ]
                n_keep = max(2, int(len(samples_with_oracle) * args.oracle_coverage))
                keep_indices = set(
                    random.sample(
                        samples_with_oracle, min(n_keep, len(samples_with_oracle))
                    )
                )

                # Store original oracle labels for visualization later
                original_oracle_labels = {}
                for i, sample in enumerate(dataset.samples):
                    if i not in keep_indices and args.oracle_field in sample.metadata:
                        original_oracle_labels[i] = sample.metadata[args.oracle_field]
                        sample.metadata[args.oracle_field] = None

            # Calibrate with cross-fitting for DR
            calibrated_dataset, cal_result = calibrate_dataset(
                dataset,
                judge_field=args.judge_field,
                oracle_field=args.oracle_field,
                enable_cross_fit=True,
                n_folds=5,
            )
            print(f"   ✓ Calibrated using {cal_result.n_oracle} oracle labels")
            print(f"   ✓ Calibration RMSE: {cal_result.calibration_rmse:.3f}")

        # Step 3: Run estimation
        print(f"\n3. Running CJE estimation with {args.estimator}...")
        sampler = PrecomputedSampler(calibrated_dataset)
        estimator = setup_estimator(args, sampler, calibrated_dataset, cal_result)
        results = estimator.fit_and_estimate()

        # Step 4: Display results
        summary_data = display_results(
            results, calibrated_dataset, sampler, estimator, args, dataset
        )

        # Step 5: Weight diagnostics
        all_weight_diagnostics = display_weight_diagnostics(
            estimator, sampler, calibrated_dataset, args
        )

        # Step 6: DR diagnostics (if applicable)
        display_dr_diagnostics(results, args)

        # Step 7: Extreme weights analysis
        analyze_extreme_weights_report(estimator, sampler, calibrated_dataset, args)

        # Restore oracle labels for visualization if they were masked
        if args.oracle_coverage < 1.0 and "original_oracle_labels" in locals():
            for i, oracle_value in original_oracle_labels.items():
                dataset.samples[i].metadata[args.oracle_field] = oracle_value

        # Step 8: Generate visualizations
        generate_visualizations(
            results,
            dataset,
            calibrated_dataset,
            estimator,
            sampler,
            args,
            summary_data,
            cal_result,
        )

        # Save results if requested
        if args.output:
            # Prepare output data
            best_diag = all_weight_diagnostics.get(summary_data["best_policy"])

            output_data = {
                "timestamp": datetime.now().isoformat(),
                "dataset": {
                    "path": args.data,
                    "n_samples": dataset.n_samples,
                    "target_policies": dataset.target_policies,
                },
                "estimation": {
                    "estimator": args.estimator,
                    "estimator_config": args.estimator_config,
                    "policies": {},
                },
                "best_policy": summary_data["best_policy"],
                "weight_diagnostics": {
                    "best_policy": {
                        "mean_weight": (
                            float(best_diag.mean_weight) if best_diag else 1.0
                        ),
                        "max_weight": float(best_diag.max_weight) if best_diag else 1.0,
                    }
                },
            }

            # Add policy results
            output_data["estimation"]["policies"]["base"] = {
                "estimate": float(summary_data["base_mean"]),
                "standard_error": float(summary_data["base_se"]),
                "ci_lower": float(summary_data["base_ci_lower"]),
                "ci_upper": float(summary_data["base_ci_upper"]),
            }

            ci_lower, ci_upper = results.confidence_interval(alpha=0.05)
            for policy, estimate, se, ci_l, ci_u in zip(
                summary_data["target_policies"],
                results.estimates,
                results.standard_errors,
                ci_lower,
                ci_upper,
            ):
                output_data["estimation"]["policies"][policy] = {
                    "estimate": float(estimate),
                    "standard_error": float(se),
                    "ci_lower": float(ci_l),
                    "ci_upper": float(ci_u),
                }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\n✓ Results written to: {args.output}")

        # Final success message
        steps_completed = 7
        if args.estimator in ["dr-cpo", "mrdr", "tmle"]:
            steps_completed += 1
        if VIZ_AVAILABLE and not args.no_plots:
            steps_completed += 1

        print(f"\n✓ Analysis complete! ({steps_completed} steps)")
        return 0

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
