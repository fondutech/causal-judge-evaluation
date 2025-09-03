#!/usr/bin/env python3
"""
Regenerate ablation plots from existing results without re-running experiments.

Usage:
    python regenerate_plots.py              # Regenerate all plots
    python regenerate_plots.py iic_effect   # Regenerate specific ablation
    python regenerate_plots.py --list       # List available ablations
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Import ablation modules
from sample_size import SampleSizeAblation
from estimator_comparison import EstimatorComparison
from interaction import InteractionAblation

# Check if these exist
try:
    from iic_effect import IICEffectAblation

    HAS_IIC = True
except ImportError:
    HAS_IIC = False

# Fresh draws ablation has been removed
HAS_FRESH = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(ablation_name: str) -> List[Dict[str, Any]]:
    """Load saved results for an ablation."""
    results_path = Path(f"results/{ablation_name}/results.jsonl")

    if not results_path.exists():
        # Try alternative paths
        alt_paths = [
            Path(f"results/{ablation_name}.jsonl"),
            Path(f"results/{ablation_name}/scenario_*.jsonl"),
        ]
        for alt in alt_paths:
            if alt.exists():
                results_path = alt
                break
            # Check glob pattern
            matches = list(Path("results").glob(alt.name))
            if matches:
                results_path = matches[0]
                break

    if not results_path.exists():
        logger.error(f"No results found for {ablation_name} at {results_path}")
        logger.info(
            f"Available results: {list(Path('results').glob('*/results.jsonl'))}"
        )
        return []

    results = []
    with open(results_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results


def regenerate_plot(ablation_name: str) -> None:
    """Regenerate plot for a specific ablation."""

    # Map ablation names to classes
    ablation_map = {
        "sample_size": SampleSizeAblation,
        "estimator_comparison": EstimatorComparison,
        "interaction": InteractionAblation,
    }

    if HAS_IIC:
        ablation_map["iic_effect"] = IICEffectAblation
    # Fresh draws ablation has been removed

    if ablation_name not in ablation_map:
        logger.error(f"Unknown ablation: {ablation_name}")
        logger.info(f"Available ablations: {list(ablation_map.keys())}")
        return

    # Load results
    results = load_results(ablation_name)
    if not results:
        logger.error(f"No results to plot for {ablation_name}")
        return

    # Create ablation instance and regenerate plot
    logger.info(f"Regenerating plot for {ablation_name}...")
    ablation = ablation_map[ablation_name]()

    # Call analyze_results method to regenerate plots
    ablation.analyze_results(results)
    logger.info(f"Plot regenerated successfully for {ablation_name}")


def list_available_ablations() -> None:
    """List all available ablations with results."""
    results_dir = Path("results")
    if not results_dir.exists():
        logger.error("No results directory found")
        return

    print("\nAvailable ablations with saved results:")
    print("-" * 40)

    for subdir in results_dir.iterdir():
        if subdir.is_dir():
            # Check for results.jsonl or other result files
            result_files = list(subdir.glob("*.jsonl")) + list(subdir.glob("*.json"))
            if result_files:
                num_files = len(result_files)
                total_results = 0
                for rf in result_files:
                    if rf.suffix == ".jsonl":
                        with open(rf) as f:
                            total_results += sum(1 for _ in f)
                    else:
                        total_results += 1

                print(
                    f"  {subdir.name:25} ({num_files} files, {total_results} results)"
                )

                # Check for plot files
                plot_files = list(subdir.glob("*.png")) + list(subdir.glob("*.pdf"))
                if plot_files:
                    print(
                        f"    Existing plots: {', '.join(p.name for p in plot_files[:3])}"
                    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate ablation plots from existing results"
    )
    parser.add_argument(
        "ablation",
        nargs="?",
        help="Name of ablation to regenerate (or 'all' for all ablations)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available ablations with results",
    )

    args = parser.parse_args()

    if args.list:
        list_available_ablations()
        return

    if not args.ablation:
        print(__doc__)
        list_available_ablations()
        return

    if args.ablation == "all":
        # Regenerate all available ablations
        ablations = ["sample_size", "estimator_comparison", "interaction"]
        if HAS_IIC:
            ablations.append("iic_effect")

        for abl in ablations:
            try:
                regenerate_plot(abl)
            except Exception as e:
                logger.error(f"Failed to regenerate {abl}: {e}")
    else:
        regenerate_plot(args.ablation)


if __name__ == "__main__":
    main()
