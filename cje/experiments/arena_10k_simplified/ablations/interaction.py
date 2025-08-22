#!/usr/bin/env python3
"""Interaction ablation - 2D grid of oracle coverage × sample size.

This ablation explores the joint effects of oracle coverage and sample size.

Key findings we expect:
- More oracle data helps more when sample size is large
- Small samples need higher oracle coverage for stability
- Trade-off frontier: you can compensate for less oracle with more samples
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append(str(Path(__file__).parent.parent))

from core import ExperimentSpec
from core.base import BaseAblation
from core.schemas import aggregate_results

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class InteractionAblation(BaseAblation):
    """Ablation to study oracle×sample interaction."""

    def __init__(self):
        super().__init__(name="interaction")

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run 2D grid of oracle coverage × sample size."""

        # Define grid
        oracle_coverages = [0.02, 0.05, 0.10, 0.20]  # 4 levels
        sample_sizes = [100, 250, 500, 1000, 2500]  # 5 levels

        # Fixed parameters
        estimator = "mrdr"  # Use best method
        n_seeds = 3  # Fewer seeds since we have many configs

        logger.info("=" * 70)
        logger.info("INTERACTION ABLATION (Oracle × Sample Size)")
        logger.info("=" * 70)
        logger.info(f"Oracle coverages: {oracle_coverages}")
        logger.info(f"Sample sizes: {sample_sizes}")
        logger.info(f"Estimator: {estimator}")
        logger.info(
            f"Total configurations: {len(oracle_coverages) * len(sample_sizes)}"
        )
        logger.info(f"Seeds per config: {n_seeds}")
        logger.info("")

        all_results = []
        config_count = 0
        total_configs = len(oracle_coverages) * len(sample_sizes)

        for oracle_coverage in oracle_coverages:
            for sample_size in sample_sizes:
                config_count += 1

                logger.info(f"\n{'='*60}")
                logger.info(
                    f"Config {config_count}/{total_configs}: "
                    f"Oracle={oracle_coverage:.0%}, n={sample_size}"
                )
                logger.info(f"{'='*60}")

                # Determine correct data path based on current directory
                data_path = Path("../data/cje_dataset.jsonl")
                if not data_path.exists():
                    data_path = Path("../../data/cje_dataset.jsonl")

                spec = ExperimentSpec(
                    ablation="interaction",
                    dataset_path=str(data_path),
                    estimator=estimator,
                    oracle_coverage=oracle_coverage,
                    sample_size=sample_size,
                    n_seeds=n_seeds,
                    seed_base=42,
                )

                # Run with multiple seeds
                results = self.run_with_seeds(spec)
                all_results.extend(results)

                # Show summary
                agg = aggregate_results(results)
                if agg.get("n_seeds_successful", 0) > 0:
                    successful = [r for r in results if r.get("success", False)]
                    mean_rmse = np.mean(
                        [r.get("rmse_vs_oracle", np.nan) for r in successful]
                    )

                    logger.info(
                        f"Results ({agg['n_seeds_successful']}/{agg['n_seeds_total']} successful):"
                    )
                    logger.info(f"  Mean RMSE: {mean_rmse:.4f}")
                    logger.info(f"  N oracle: {oracle_coverage * sample_size:.0f}")
                else:
                    logger.warning(f"All {agg['n_seeds_total']} seeds failed!")

        # Save all results
        output_dir = Path("results/interaction")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy types before saving
        def convert_numpy(obj):
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.bool8)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif hasattr(obj, "item"):
                return obj.item()
            return obj

        with open(output_dir / "results.jsonl", "w") as f:
            for result in all_results:
                f.write(json.dumps(convert_numpy(result)) + "\n")

        logger.info(f"\nSaved {len(all_results)} results to {output_dir}")

        return all_results

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interaction effects."""

        # Create 2D grid of RMSE values
        rmse_grid = {}
        success_grid = {}

        for r in results:
            if r.get("success", False):
                oracle = r["spec"]["oracle_coverage"]
                n_samples = r["spec"]["sample_size"]
                key = (oracle, n_samples)

                if key not in rmse_grid:
                    rmse_grid[key] = []
                    success_grid[key] = 0

                rmse_grid[key].append(r.get("rmse_vs_oracle", np.nan))
                success_grid[key] += 1

        # Average across seeds
        mean_rmse_grid = {}
        for key, rmses in rmse_grid.items():
            mean_rmse_grid[key] = np.nanmean(rmses)

        # Extract unique values for axes
        oracle_values = sorted(set(k[0] for k in mean_rmse_grid.keys()))
        sample_values = sorted(set(k[1] for k in mean_rmse_grid.keys()))

        # Create matrix for heatmap
        rmse_matrix = np.full((len(oracle_values), len(sample_values)), np.nan)
        for i, oracle in enumerate(oracle_values):
            for j, n_samples in enumerate(sample_values):
                if (oracle, n_samples) in mean_rmse_grid:
                    rmse_matrix[i, j] = mean_rmse_grid[(oracle, n_samples)]

        # Find sweet spots (good trade-offs)
        sweet_spots = []
        threshold = np.nanquantile(rmse_matrix, 0.25)  # Top 25% performance

        for i, oracle in enumerate(oracle_values):
            for j, n_samples in enumerate(sample_values):
                if rmse_matrix[i, j] <= threshold:
                    n_oracle = oracle * n_samples
                    efficiency = 1.0 / (
                        n_oracle * rmse_matrix[i, j]
                    )  # Higher is better
                    sweet_spots.append(
                        {
                            "oracle_coverage": oracle,
                            "sample_size": n_samples,
                            "n_oracle": n_oracle,
                            "rmse": rmse_matrix[i, j],
                            "efficiency": efficiency,
                        }
                    )

        # Sort by efficiency
        sweet_spots.sort(key=lambda x: x["efficiency"], reverse=True)

        return {
            "oracle_values": oracle_values,
            "sample_values": sample_values,
            "rmse_matrix": rmse_matrix,
            "mean_rmse_grid": mean_rmse_grid,
            "success_grid": success_grid,
            "sweet_spots": sweet_spots[:5],  # Top 5 trade-offs
        }

    def create_figure(self, results: List[Dict[str, Any]], output_path: Path = None):
        """Create Figure 3: Interaction heatmap."""

        analysis = self.analyze_results(results)

        if not analysis["oracle_values"] or not analysis["sample_values"]:
            logger.warning("No successful results to plot")
            return

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Convert to percentages for display
        oracle_labels = [f"{x:.0%}" for x in analysis["oracle_values"]]
        sample_labels = [str(x) for x in analysis["sample_values"]]

        # Panel A: RMSE heatmap
        ax = axes[0]
        sns.heatmap(
            analysis["rmse_matrix"],
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",  # Red=bad, Green=good
            xticklabels=sample_labels,
            yticklabels=oracle_labels,
            cbar_kws={"label": "RMSE"},
            ax=ax,
        )
        ax.set_xlabel("Sample Size", fontsize=12)
        ax.set_ylabel("Oracle Coverage", fontsize=12)
        ax.set_title("A. RMSE vs Oracle Truth", fontsize=14, fontweight="bold")

        # Panel B: Efficiency contours (1 / (n_oracle * RMSE))
        ax = axes[1]

        # Compute efficiency matrix
        efficiency_matrix = np.zeros_like(analysis["rmse_matrix"])
        for i, oracle in enumerate(analysis["oracle_values"]):
            for j, n_samples in enumerate(analysis["sample_values"]):
                n_oracle = oracle * n_samples
                if np.isfinite(analysis["rmse_matrix"][i, j]):
                    efficiency_matrix[i, j] = 1000.0 / (
                        n_oracle * analysis["rmse_matrix"][i, j]
                    )
                else:
                    efficiency_matrix[i, j] = np.nan

        # Create contour plot
        X, Y = np.meshgrid(range(len(sample_labels)), range(len(oracle_labels)))
        contour = ax.contour(
            X, Y, efficiency_matrix, levels=5, colors="black", alpha=0.4
        )
        ax.clabel(contour, inline=True, fontsize=10)

        im = ax.contourf(X, Y, efficiency_matrix, levels=10, cmap="viridis")
        plt.colorbar(im, ax=ax, label="Efficiency Score")

        # Mark sweet spots
        for spot in analysis["sweet_spots"][:3]:  # Top 3
            i = analysis["oracle_values"].index(spot["oracle_coverage"])
            j = analysis["sample_values"].index(spot["sample_size"])
            ax.plot(j, i, "r*", markersize=15)

        ax.set_xticks(range(len(sample_labels)))
        ax.set_xticklabels(sample_labels)
        ax.set_yticks(range(len(oracle_labels)))
        ax.set_yticklabels(oracle_labels)
        ax.set_xlabel("Sample Size", fontsize=12)
        ax.set_ylabel("Oracle Coverage", fontsize=12)
        ax.set_title(
            "B. Efficiency (Lower Cost × Error)", fontsize=14, fontweight="bold"
        )

        plt.suptitle(
            "Oracle × Sample Size Interaction", fontsize=16, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {output_path}")

        plt.show()

        return fig


def main():
    """Run interaction ablation."""

    ablation = InteractionAblation()

    # Run the ablation
    results = ablation.run_ablation()

    # Analyze
    analysis = ablation.analyze_results(results)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    if analysis["sweet_spots"]:
        logger.info("\nTop efficiency configurations:")
        logger.info("(Balancing oracle cost and RMSE)")

        for i, spot in enumerate(analysis["sweet_spots"], 1):
            logger.info(
                f"\n{i}. Oracle={spot['oracle_coverage']:.0%}, n={spot['sample_size']}"
            )
            logger.info(f"   N oracle labels: {spot['n_oracle']:.0f}")
            logger.info(f"   RMSE: {spot['rmse']:.4f}")
            logger.info(f"   Efficiency score: {spot['efficiency']:.2f}")

    # Show overall patterns
    if len(analysis["rmse_matrix"]) > 0:
        logger.info("\nOverall patterns:")

        # Effect of doubling oracle coverage
        if len(analysis["oracle_values"]) >= 2:
            first_col = analysis["rmse_matrix"][:, 0]  # First sample size
            valid = np.isfinite(first_col)
            if np.sum(valid) >= 2:
                improvement = (first_col[valid][0] - first_col[valid][-1]) / first_col[
                    valid
                ][0]
                logger.info(
                    f"  Increasing oracle 10x: ~{improvement:.0%} RMSE reduction"
                )

        # Effect of doubling sample size
        if len(analysis["sample_values"]) >= 2:
            first_row = analysis["rmse_matrix"][0, :]  # First oracle coverage
            valid = np.isfinite(first_row)
            if np.sum(valid) >= 2:
                improvement = (first_row[valid][0] - first_row[valid][-1]) / first_row[
                    valid
                ][0]
                logger.info(
                    f"  Increasing samples 25x: ~{improvement:.0%} RMSE reduction"
                )

    # Create figure
    figure_path = Path("results/interaction/figure_3_interaction.png")
    ablation.create_figure(results, figure_path)

    logger.info("\n" + "=" * 70)
    logger.info("INTERACTION ABLATION COMPLETE")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
