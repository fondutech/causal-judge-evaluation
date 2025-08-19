#!/usr/bin/env python3
"""Run small-scale versions of all three core ablations for testing.

This script runs reduced versions of the ablations to verify everything works
before launching the full experiments.
"""

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "experiments"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_oracle_coverage_small():
    """Run oracle coverage with fewer points."""
    logger.info("\n" + "=" * 70)
    logger.info("ORACLE COVERAGE ABLATION (SMALL SCALE)")
    logger.info("=" * 70)

    from oracle_coverage import OracleCoverageAblation
    from core import ExperimentSpec

    ablation = OracleCoverageAblation()
    all_results = []

    # Test just 3 coverage levels with small data
    coverages = [0.05, 0.10, 0.20]

    for coverage in coverages:
        spec = ExperimentSpec(
            ablation="oracle_coverage_small",
            dataset_path="../data/cje_dataset.jsonl",
            estimator="calibrated-ips",  # Faster than MRDR
            oracle_coverage=coverage,
            sample_fraction=0.1,  # 10% of data (~500 samples)
            n_seeds=2,
            seed_base=42,
        )

        logger.info(f"\nTesting oracle coverage = {coverage:.0%}")
        results = ablation.run_with_seeds(spec)
        all_results.extend(results)

    # Quick analysis
    analysis = ablation.analyze_results(all_results)
    logger.info("\nResults summary:")
    for cov in coverages:
        if cov in analysis["mean_rmse"]:
            logger.info(f"  {cov:5.0%}: RMSE = {analysis['mean_rmse'][cov]:.4f}")

    return all_results


def run_sample_size_small():
    """Run sample size with fewer points."""
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE SIZE ABLATION (SMALL SCALE)")
    logger.info("=" * 70)

    from sample_size import SampleSizeAblation
    from core import ExperimentSpec

    ablation = SampleSizeAblation()
    all_results = []

    # Test just 3 sample sizes
    sample_sizes = [100, 500, 1000]

    for n_samples in sample_sizes:
        spec = ExperimentSpec(
            ablation="sample_size_small",
            dataset_path="../data/cje_dataset.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=0.10,  # Fixed 10%
            sample_size=n_samples,
            n_seeds=2,
            seed_base=42,
        )

        logger.info(f"\nTesting n = {n_samples}")
        results = ablation.run_with_seeds(spec)
        all_results.extend(results)

    # Quick analysis
    analysis = ablation.analyze_results(all_results)
    logger.info("\nResults summary:")
    if "calibrated-ips" in analysis:
        for i, n in enumerate(analysis["calibrated-ips"]["sample_sizes"]):
            rmse = analysis["calibrated-ips"]["mean_rmse"][i]
            logger.info(f"  n={n:4d}: RMSE = {rmse:.4f}")

    return all_results


def run_interaction_small():
    """Run interaction with small 2x3 grid."""
    logger.info("\n" + "=" * 70)
    logger.info("INTERACTION ABLATION (SMALL SCALE)")
    logger.info("=" * 70)

    from interaction import InteractionAblation
    from core import ExperimentSpec

    ablation = InteractionAblation()
    all_results = []

    # Small 2x3 grid
    oracle_coverages = [0.05, 0.10]
    sample_sizes = [100, 500, 1000]

    logger.info(f"Testing {len(oracle_coverages)}x{len(sample_sizes)} grid")

    for oracle in oracle_coverages:
        for n_samples in sample_sizes:
            spec = ExperimentSpec(
                ablation="interaction_small",
                dataset_path="../data/cje_dataset.jsonl",
                estimator="calibrated-ips",
                oracle_coverage=oracle,
                sample_size=n_samples,
                n_seeds=1,  # Just 1 seed for grid
                seed_base=42,
            )

            logger.info(f"\nTesting oracle={oracle:.0%}, n={n_samples}")
            results = ablation.run_with_seeds(spec)
            all_results.extend(results)

    # Quick analysis
    analysis = ablation.analyze_results(all_results)
    logger.info("\nGrid summary:")
    for oracle in oracle_coverages:
        for n_samples in sample_sizes:
            key = (oracle, n_samples)
            if key in analysis["mean_rmse_grid"]:
                rmse = analysis["mean_rmse_grid"][key]
                logger.info(
                    f"  Oracle={oracle:5.0%}, n={n_samples:4d}: RMSE = {rmse:.4f}"
                )

    return all_results


def main():
    """Run all small-scale ablations."""

    logger.info("=" * 70)
    logger.info("RUNNING SMALL-SCALE ABLATION TESTS")
    logger.info("=" * 70)
    logger.info("\nThis will test all three core ablations with reduced data.")
    logger.info("If successful, run the full ablations with run_full_scale.py")

    try:
        # Oracle coverage
        oracle_results = run_oracle_coverage_small()
        logger.info(f"\n✓ Oracle coverage: {len(oracle_results)} results")

        # Sample size
        sample_results = run_sample_size_small()
        logger.info(f"\n✓ Sample size: {len(sample_results)} results")

        # Interaction
        interaction_results = run_interaction_small()
        logger.info(f"\n✓ Interaction: {len(interaction_results)} results")

        logger.info("\n" + "=" * 70)
        logger.info("SMALL-SCALE TESTS COMPLETE")
        logger.info("=" * 70)
        logger.info("\n✓ All ablations working!")
        logger.info("Ready to run full experiments with run_full_scale.py")

        return True

    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
