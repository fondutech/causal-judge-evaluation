#!/usr/bin/env python3
"""
Experiment comparing StackedDREstimator with other estimators on Arena data.

This will compare:
1. CalibratedIPS (baseline, no fresh draws needed)
2. DR-CPO (single DR method)
3. TMLE (single DR method)
4. MRDR (single DR method)
5. StackedDR (combination of all DR methods)
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add parent to path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.data import PrecomputedSampler
from cje.estimators import CalibratedIPS, StackedDREstimator
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.tmle import TMLEEstimator
from cje.estimators.mrdr import MRDREstimator
from cje.data.fresh_draws import load_fresh_draws_auto


def run_estimator(
    name: str, estimator, sampler, fresh_draws_dir=None
) -> Dict[str, Any]:
    """Run a single estimator and collect results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {name}...")
    logger.info(f"{'='*60}")

    start_time = time.time()

    try:
        # Add fresh draws if needed and available
        if fresh_draws_dir and hasattr(estimator, "add_fresh_draws"):
            for policy in sampler.target_policies:
                try:
                    fresh_draws = load_fresh_draws_auto(
                        Path(fresh_draws_dir), policy, verbose=False
                    )
                    estimator.add_fresh_draws(policy, fresh_draws)
                    logger.info(f"  Added fresh draws for {policy}")
                except FileNotFoundError:
                    logger.warning(f"  No fresh draws found for {policy}")

        # Run estimation
        result = estimator.fit_and_estimate()
        elapsed = time.time() - start_time

        # Collect results
        estimates = {}
        standard_errors = {}
        for i, policy in enumerate(sampler.target_policies):
            estimates[policy] = float(result.estimates[i])
            standard_errors[policy] = float(result.standard_errors[i])

        # Get diagnostics
        diagnostics = {}
        if result.diagnostics:
            if hasattr(result.diagnostics, "weight_ess"):
                diagnostics["ess"] = float(result.diagnostics.weight_ess)
            if hasattr(result.diagnostics, "weight_status"):
                diagnostics["status"] = str(result.diagnostics.weight_status)

        # Get stacking-specific info
        stacking_info = {}
        if name == "StackedDR" and result.metadata:
            if "stacking_weights" in result.metadata:
                stacking_info["weights"] = {
                    policy: {
                        est: float(w)
                        for est, w in zip(
                            result.metadata.get("valid_estimators", []), weights
                        )
                    }
                    for policy, weights in result.metadata["stacking_weights"].items()
                }
            if "valid_estimators" in result.metadata:
                stacking_info["valid_estimators"] = result.metadata["valid_estimators"]
            if "failed_estimators" in result.metadata:
                stacking_info["failed_estimators"] = result.metadata[
                    "failed_estimators"
                ]

        return {
            "success": True,
            "estimates": estimates,
            "standard_errors": standard_errors,
            "elapsed_time": elapsed,
            "diagnostics": diagnostics,
            "stacking_info": stacking_info,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"  Failed: {e}")
        return {"success": False, "error": str(e), "elapsed_time": elapsed}


def main():
    """Run comparison experiments."""

    # Configuration
    DATA_PATH = "cje/experiments/arena_10k_simplified/data/cje_dataset.jsonl"
    FRESH_DRAWS_DIR = "cje/experiments/arena_10k_simplified/data/responses"
    ORACLE_COVERAGE = 0.5  # Use 50% of oracle labels for calibration
    SEED = 42

    # Set random seed
    np.random.seed(SEED)

    logger.info("=" * 80)
    logger.info("STACKING ESTIMATOR COMPARISON EXPERIMENT")
    logger.info("=" * 80)

    # Load and prepare data
    logger.info("\n1. Loading and calibrating dataset...")
    dataset = load_dataset_from_jsonl(DATA_PATH)
    logger.info(f"   Loaded {len(dataset.samples)} samples")
    logger.info(f"   Target policies: {dataset.target_policies}")

    # Simulate partial oracle coverage for calibration
    if ORACLE_COVERAGE < 1.0:
        logger.info(f"   Masking oracle labels to {ORACLE_COVERAGE*100:.0f}% coverage")
        samples_with_oracle = [
            i
            for i, s in enumerate(dataset.samples)
            if "oracle_label" in s.metadata and s.metadata["oracle_label"] is not None
        ]
        n_keep = int(len(samples_with_oracle) * ORACLE_COVERAGE)
        keep_indices = set(np.random.choice(samples_with_oracle, n_keep, replace=False))

        for i, sample in enumerate(dataset.samples):
            if i not in keep_indices and "oracle_label" in sample.metadata:
                sample.metadata["oracle_label"] = None

    # Calibrate dataset
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True,
        n_folds=5,
    )

    sampler = PrecomputedSampler(calibrated_dataset)
    logger.info(f"   Calibration complete, {sampler.n_valid_samples} valid samples")

    # Initialize estimators
    estimators = {
        "CalibratedIPS": CalibratedIPS(sampler),
        "DR-CPO": DRCPOEstimator(
            sampler, calibrator=cal_result.calibrator if cal_result else None
        ),
        "TMLE": TMLEEstimator(
            sampler, calibrator=cal_result.calibrator if cal_result else None
        ),
        "MRDR": MRDREstimator(
            sampler, calibrator=cal_result.calibrator if cal_result else None
        ),
    }

    # Create StackedDR last and add fresh draws before component estimators run
    stacked_estimator = StackedDREstimator(
        sampler,
        estimators=["dr-cpo", "tmle", "mrdr"],
        use_outer_split=True,
        parallel=False,  # Sequential for clearer output
    )

    # Add fresh draws to the stacked estimator
    if Path(FRESH_DRAWS_DIR).exists():
        for policy in sampler.target_policies:
            try:
                fresh_draws = load_fresh_draws_auto(
                    Path(FRESH_DRAWS_DIR), policy, verbose=False
                )
                stacked_estimator.add_fresh_draws(policy, fresh_draws)
            except FileNotFoundError:
                pass

    estimators["StackedDR"] = stacked_estimator

    # Run experiments
    results = {}
    for name, estimator in estimators.items():
        # DR methods need fresh draws
        needs_fresh = name != "CalibratedIPS"
        results[name] = run_estimator(
            name, estimator, sampler, FRESH_DRAWS_DIR if needs_fresh else None
        )

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    # Estimates comparison
    logger.info("\nEstimates (with 95% CI):")
    logger.info("-" * 60)
    logger.info(f"{'Estimator':<15} {'Policy':<25} {'Estimate':<20} {'Time (s)':<10}")
    logger.info("-" * 60)

    for name, result in results.items():
        if result["success"]:
            for policy in sampler.target_policies:
                est = result["estimates"][policy]
                se = result["standard_errors"][policy]
                ci_lower = est - 1.96 * se
                ci_upper = est + 1.96 * se
                time_str = f"{result['elapsed_time']:.2f}"
                logger.info(
                    f"{name:<15} {policy:<25} {est:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]  {time_str:<10}"
                )
        else:
            logger.info(
                f"{name:<15} {'FAILED':<25} {result.get('error', 'Unknown error')[:40]}"
            )

    # Standard errors comparison
    logger.info("\n\nStandard Errors:")
    logger.info("-" * 60)
    logger.info(f"{'Estimator':<15} {'Policy':<25} {'SE':<15} {'% vs IPS':<15}")
    logger.info("-" * 60)

    ips_ses = results.get("CalibratedIPS", {}).get("standard_errors", {})
    for name, result in results.items():
        if result["success"]:
            for policy in sampler.target_policies:
                se = result["standard_errors"][policy]
                if policy in ips_ses and ips_ses[policy] > 0:
                    pct_change = ((se / ips_ses[policy]) - 1) * 100
                    pct_str = f"{pct_change:+.1f}%"
                else:
                    pct_str = "N/A"
                logger.info(f"{name:<15} {policy:<25} {se:.6f}    {pct_str:<15}")

    # Stacking weights (if available)
    if "StackedDR" in results and results["StackedDR"]["success"]:
        stacking_info = results["StackedDR"].get("stacking_info", {})
        if "weights" in stacking_info:
            logger.info("\n\nStacking Weights:")
            logger.info("-" * 60)
            for policy, weights in stacking_info["weights"].items():
                logger.info(f"\n{policy}:")
                for est, weight in weights.items():
                    logger.info(f"  {est:<10}: {weight:.3f}")

        if "valid_estimators" in stacking_info:
            logger.info(
                f"\nValid estimators: {', '.join(stacking_info['valid_estimators'])}"
            )
        if "failed_estimators" in stacking_info:
            logger.info(
                f"Failed estimators: {', '.join(stacking_info['failed_estimators'])}"
            )

    # Variance reduction analysis
    logger.info("\n\nVariance Reduction (vs CalibratedIPS):")
    logger.info("-" * 60)

    if "CalibratedIPS" in results and results["CalibratedIPS"]["success"]:
        for name in ["DR-CPO", "TMLE", "MRDR", "StackedDR"]:
            if name in results and results[name]["success"]:
                logger.info(f"\n{name}:")
                for policy in sampler.target_policies:
                    ips_var = ips_ses[policy] ** 2
                    dr_var = results[name]["standard_errors"][policy] ** 2
                    var_reduction = (1 - dr_var / ips_var) * 100
                    logger.info(f"  {policy}: {var_reduction:+.1f}% variance reduction")

    # Save results to JSON
    output_file = "stacking_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n\nResults saved to {output_file}")

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
