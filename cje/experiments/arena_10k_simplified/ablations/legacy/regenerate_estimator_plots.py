#!/usr/bin/env python3
"""
Regenerate just the estimator comparison plots from existing results.

This is a focused script for the most common use case - regenerating
the estimator comparison visualizations without re-running experiments.

Usage:
    python regenerate_estimator_plots.py
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from estimator_comparison import EstimatorComparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    """Regenerate estimator comparison plots from saved results."""

    # Check for existing results
    results_path = Path("results/estimator_comparison/results.jsonl")

    if not results_path.exists():
        logger.error(f"No results found at {results_path}")
        logger.info("Please run the estimator comparison ablation first:")
        logger.info("  python estimator_comparison.py")
        return 1

    # Load results
    results = []
    with open(results_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    logger.info(f"Loaded {len(results)} results from {results_path}")

    if not results:
        logger.error("No results found in file")
        return 1

    # Create ablation instance
    ablation = EstimatorComparison()

    # Regenerate all plots
    logger.info("Regenerating estimator comparison plots...")
    ablation.analyze_results(results)

    # List generated files
    output_dir = Path("results/estimator_comparison")
    plots = list(output_dir.glob("*.png")) + list(output_dir.glob("*.pdf"))

    if plots:
        logger.info(f"\nGenerated {len(plots)} plots:")
        for plot in sorted(plots):
            logger.info(f"  - {plot.name}")

    logger.info("\nPlots regenerated successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
