#!/usr/bin/env python3
"""Run full-scale versions of the three core ablations.

This runs the complete experiments as designed for the paper.
Expect this to take several hours.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "experiments"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run all full-scale ablations."""

    start_time = time.time()

    logger.info("=" * 70)
    logger.info("RUNNING FULL-SCALE ABLATIONS")
    logger.info("=" * 70)
    logger.info("\nThis will run the complete ablation experiments.")
    logger.info("Expected runtime: 2-4 hours")
    logger.info("")

    # Import ablations
    from oracle_coverage import main as run_oracle_coverage
    from sample_size import main as run_sample_size
    from interaction import main as run_interaction

    results = {}

    try:
        # 1. Oracle coverage - THE fundamental result
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION 1/3: ORACLE COVERAGE")
        logger.info("=" * 70)
        oracle_results = run_oracle_coverage()
        results["oracle_coverage"] = oracle_results
        logger.info(f"✓ Oracle coverage complete: {len(oracle_results)} results")

        # 2. Sample size - Classic scaling analysis
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION 2/3: SAMPLE SIZE")
        logger.info("=" * 70)
        sample_results = run_sample_size()
        results["sample_size"] = sample_results
        logger.info(f"✓ Sample size complete: {len(sample_results)} results")

        # 3. Interaction - 2D exploration
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION 3/3: INTERACTION")
        logger.info("=" * 70)
        interaction_results = run_interaction()
        results["interaction"] = interaction_results
        logger.info(f"✓ Interaction complete: {len(interaction_results)} results")

        # Summary
        runtime = time.time() - start_time
        logger.info("\n" + "=" * 70)
        logger.info("ALL ABLATIONS COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nTotal runtime: {runtime/3600:.1f} hours")
        logger.info(f"Total experiments: {sum(len(r) for r in results.values())}")

        logger.info("\nResults saved to:")
        logger.info("  ablations/results/oracle_coverage/")
        logger.info("  ablations/results/sample_size/")
        logger.info("  ablations/results/interaction/")

        logger.info("\nFigures saved to:")
        logger.info("  ablations/results/oracle_coverage/figure_1_oracle_coverage.png")
        logger.info("  ablations/results/sample_size/figure_2_sample_scaling.png")
        logger.info("  ablations/results/interaction/figure_3_interaction.png")

        return True

    except Exception as e:
        logger.error(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
