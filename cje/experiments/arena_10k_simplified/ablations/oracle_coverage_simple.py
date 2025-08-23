#!/usr/bin/env python3
"""Simplified oracle coverage ablation - testing how much oracle data we need.

This version removes caching complexity while keeping the core experiment logic.
"""

import json
import logging
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt

# Add parent dirs to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.mrdr import MRDREstimator
from cje.data.fresh_draws import load_fresh_draws_auto

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def mask_oracle_labels(dataset, oracle_coverage: float, seed: int):
    """Mask oracle labels to simulate partial coverage.

    Args:
        dataset: Dataset with samples
        oracle_coverage: Fraction to keep (0-1)
        seed: Random seed for reproducibility

    Returns:
        Number of oracle labels kept
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find samples with oracle labels
    oracle_indices = [
        i
        for i, s in enumerate(dataset.samples)
        if s.metadata.get("oracle_label") is not None
    ]

    # Keep only specified fraction
    n_keep = max(2, int(len(oracle_indices) * oracle_coverage))
    keep_indices = set(random.sample(oracle_indices, min(n_keep, len(oracle_indices))))

    # Mask the rest
    for i, sample in enumerate(dataset.samples):
        if i not in keep_indices and "oracle_label" in sample.metadata:
            sample.metadata["oracle_label"] = None

    return len(keep_indices)


def run_single_experiment(
    dataset_path: str, oracle_coverage: float, seed: int
) -> Dict[str, Any]:
    """Run a single oracle coverage experiment.

    Args:
        dataset_path: Path to dataset
        oracle_coverage: Fraction of oracle labels to use
        seed: Random seed

    Returns:
        Dictionary with results
    """
    result = {"oracle_coverage": oracle_coverage, "seed": seed, "success": False}

    try:
        # Load dataset
        dataset = load_dataset_from_jsonl(dataset_path)
        n_samples = len(dataset.samples)

        # Mask oracle labels
        n_oracle = mask_oracle_labels(dataset, oracle_coverage, seed)
        result["n_samples"] = n_samples
        result["n_oracle"] = n_oracle

        # Calibrate dataset
        calibrated_dataset, cal_result = calibrate_dataset(
            dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        if cal_result:
            result["calibration_rmse"] = float(cal_result.calibration_rmse)

        # Create estimator (MRDR)
        sampler = PrecomputedSampler(calibrated_dataset)
        estimator = MRDREstimator(
            sampler,
            calibrator=cal_result.calibrator if cal_result else None,
            n_folds=5,
            oracle_slice_config=(oracle_coverage < 1.0),
        )

        # Add fresh draws
        data_dir = Path(dataset_path).parent
        for policy in sampler.target_policies:
            try:
                fresh_draws = load_fresh_draws_auto(data_dir, policy, verbose=False)
                estimator.add_fresh_draws(policy, fresh_draws)
            except FileNotFoundError:
                logger.warning(f"No fresh draws for {policy}")

        # Run estimation
        estimation_result = estimator.fit_and_estimate()

        # Extract results
        result["estimates"] = {}
        result["standard_errors"] = {}

        for i, policy in enumerate(sampler.target_policies):
            result["estimates"][policy] = float(estimation_result.estimates[i])
            result["standard_errors"][policy] = float(
                estimation_result.standard_errors[i]
            )

        # Add diagnostics if available
        if estimation_result.diagnostics:
            diag = estimation_result.diagnostics
            result["mean_ess"] = float(diag.weight_ess) if diag.weight_ess else None

        result["success"] = True

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        result["error"] = str(e)

    return result


def compute_oracle_ground_truth(dataset_path: str) -> Dict[str, float]:
    """Load oracle ground truth values from response files.

    Args:
        dataset_path: Path to dataset file

    Returns:
        Dictionary mapping policy names to oracle means
    """
    oracle_means = {}
    data_dir = Path(dataset_path).parent
    responses_dir = data_dir / "responses"

    # Get policy names from a sample file or hardcode the known ones
    policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]

    for policy in policies:
        response_file = responses_dir / f"{policy}_responses.jsonl"
        if response_file.exists():
            oracle_values = []
            with open(response_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "metadata" in data and "oracle_label" in data["metadata"]:
                            oracle_val = data["metadata"]["oracle_label"]
                            if oracle_val is not None:
                                oracle_values.append(oracle_val)
                    except json.JSONDecodeError:
                        continue

            if oracle_values:
                oracle_means[policy] = float(np.mean(oracle_values))

    return oracle_means


def main():
    """Run oracle coverage ablation."""

    # Configuration
    oracle_coverages = [0.05, 0.10]  # Just test 2 levels for now
    n_seeds = 1  # Just 1 seed for testing

    # Find dataset
    data_path = Path("../data/cje_dataset.jsonl")
    if not data_path.exists():
        data_path = Path("../../data/cje_dataset.jsonl")

    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        return

    logger.info("=" * 70)
    logger.info("ORACLE COVERAGE ABLATION (SIMPLIFIED)")
    logger.info("=" * 70)
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Coverage levels: {oracle_coverages}")
    logger.info(f"Seeds per level: {n_seeds}")
    logger.info("")

    # Get ground truth
    oracle_truth = compute_oracle_ground_truth(str(data_path))
    logger.info(f"Oracle ground truth loaded for {len(oracle_truth)} policies")
    logger.info("")

    # Run experiments
    all_results = []

    for coverage in oracle_coverages:
        logger.info(f"\nOracle Coverage: {coverage:.0%}")
        logger.info("-" * 40)

        coverage_results = []
        for seed_offset in range(n_seeds):
            seed = 42 + seed_offset

            result = run_single_experiment(str(data_path), coverage, seed)

            if result["success"]:
                # Compute RMSE vs oracle if we have ground truth
                if oracle_truth and result["estimates"]:
                    errors = []
                    for policy, estimate in result["estimates"].items():
                        if policy in oracle_truth:
                            errors.append((estimate - oracle_truth[policy]) ** 2)
                    if errors:
                        result["rmse_vs_oracle"] = float(np.sqrt(np.mean(errors)))

                logger.info(
                    f"  Seed {seed}: ✓ RMSE={result.get('rmse_vs_oracle', 0):.4f}"
                )
            else:
                logger.info(f"  Seed {seed}: ✗ {result.get('error', 'Failed')}")

            coverage_results.append(result)
            all_results.append(result)

        # Show summary for this coverage level
        successful = [r for r in coverage_results if r["success"]]
        if successful:
            mean_rmse = np.mean([r.get("rmse_vs_oracle", np.nan) for r in successful])
            logger.info(f"  Mean RMSE: {mean_rmse:.4f}")

    # Save results
    output_dir = Path("results/oracle_coverage")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results_simple.jsonl", "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    logger.info(
        f"\nSaved {len(all_results)} results to {output_dir}/results_simple.jsonl"
    )

    # Analysis
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    # Group by coverage
    by_coverage = {}
    for r in all_results:
        if r["success"]:
            cov = r["oracle_coverage"]
            if cov not in by_coverage:
                by_coverage[cov] = []
            by_coverage[cov].append(r.get("rmse_vs_oracle", np.nan))

    logger.info("\nRMSE by Coverage:")
    for cov in sorted(by_coverage.keys()):
        rmses = by_coverage[cov]
        mean_rmse = np.nanmean(rmses)
        std_rmse = np.nanstd(rmses)
        logger.info(f"  {cov:6.1%}: {mean_rmse:.4f} ± {std_rmse:.4f}")

    # Find sweet spot (where improvement slows)
    coverages = sorted(by_coverage.keys())
    mean_rmses = [np.nanmean(by_coverage[c]) for c in coverages]
    if len(mean_rmses) > 2:
        improvements = -np.diff(mean_rmses)
        # Find first point where improvement < 10% of initial
        if improvements[0] != 0:
            threshold = 0.1 * abs(improvements[0])
            for i, imp in enumerate(improvements):
                if abs(imp) < threshold:
                    logger.info(f"\nSweet spot: {coverages[i]:.1%} coverage")
                    logger.info("(Diminishing returns beyond this point)")
                    break

    # Create simple visualization
    if by_coverage:
        plt.figure(figsize=(10, 6))

        coverages_pct = [c * 100 for c in coverages]
        plt.plot(coverages_pct, mean_rmses, "o-", linewidth=2, markersize=8)

        plt.xlabel("Oracle Coverage (%)", fontsize=12)
        plt.ylabel("RMSE vs Oracle Truth", fontsize=12)
        plt.title("Oracle Coverage Impact on Estimation Error", fontsize=14)
        plt.grid(True, alpha=0.3)

        figure_path = output_dir / "oracle_coverage_simple.png"
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        logger.info(f"\nSaved figure to {figure_path}")
        plt.close()

    logger.info("\n" + "=" * 70)
    logger.info("ORACLE COVERAGE ABLATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
