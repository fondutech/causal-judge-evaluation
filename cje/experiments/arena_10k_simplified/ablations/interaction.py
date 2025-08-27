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
        oracle_coverages = [0.02, 0.05, 0.10, 0.20, 0.50]  # 5 levels
        sample_sizes = [100, 250, 500, 1000, 2500, 5000]  # 6 levels

        # Fixed parameters
        estimator = "stacked-dr"  # Use best method
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
        """Analyze interaction effects with MDE-based sweet spots."""

        # Create 2D grids for RMSE and SE values
        rmse_grid = {}
        se_grid = {}
        success_grid = {}

        for r in results:
            if r.get("success", False):
                oracle = r["spec"]["oracle_coverage"]
                n_samples = r["spec"]["sample_size"]
                key = (oracle, n_samples)

                if key not in rmse_grid:
                    rmse_grid[key] = []
                    se_grid[key] = []
                    success_grid[key] = 0

                rmse_grid[key].append(r.get("rmse_vs_oracle", np.nan))
                
                # Collect standard errors (proper IC-based SEs)
                if "standard_errors" in r and r["standard_errors"]:
                    # Average SE across policies for this configuration
                    avg_se = np.nanmean(list(r["standard_errors"].values()))
                    se_grid[key].append(avg_se)
                else:
                    # Fallback to RMSE if SE not available
                    se_grid[key].append(r.get("rmse_vs_oracle", np.nan))
                    
                success_grid[key] += 1

        # Average across seeds
        mean_rmse_grid = {}
        mean_se_grid = {}
        for key, rmses in rmse_grid.items():
            mean_rmse_grid[key] = np.nanmean(rmses)
            mean_se_grid[key] = np.nanmean(se_grid[key])

        # Extract unique values for axes
        oracle_values = sorted(set(k[0] for k in mean_rmse_grid.keys()))
        sample_values = sorted(set(k[1] for k in mean_rmse_grid.keys()))

        # Create matrices for heatmap
        rmse_matrix = np.full((len(oracle_values), len(sample_values)), np.nan)
        se_matrix = np.full((len(oracle_values), len(sample_values)), np.nan)
        for i, oracle in enumerate(oracle_values):
            for j, n_samples in enumerate(sample_values):
                if (oracle, n_samples) in mean_rmse_grid:
                    rmse_matrix[i, j] = mean_rmse_grid[(oracle, n_samples)]
                    se_matrix[i, j] = mean_se_grid[(oracle, n_samples)]

        # Find sweet spots based on MDE thresholds using proper SEs
        alpha = 0.05
        power = 0.80
        z_alpha = 1.96
        z_power = 0.84
        k = z_alpha + z_power  # ≈ 2.80
        
        sweet_spots = []
        target_mdes = [0.01, 0.02]  # 1% and 2% effect sizes
        
        for i, oracle in enumerate(oracle_values):
            for j, n_samples in enumerate(sample_values):
                if np.isfinite(se_matrix[i, j]):
                    n_oracle = oracle * n_samples
                    
                    # MDE for two-policy comparison using proper SE
                    mde_two = k * np.sqrt(2.0) * se_matrix[i, j]
                    
                    # Check which MDE targets this config can achieve
                    achievable_targets = [t for t in target_mdes if mde_two <= t]
                    
                    if achievable_targets:
                        # Compute cost-efficiency (lower is better)
                        # Assuming unit cost per oracle label for simplicity
                        cost_per_percent_mde = n_oracle / min(achievable_targets)
                        
                        sweet_spots.append(
                            {
                                "oracle_coverage": oracle,
                                "sample_size": n_samples,
                                "n_oracle": n_oracle,
                                "rmse": rmse_matrix[i, j],
                                "mde_two": mde_two,
                                "achievable_mde": min(achievable_targets),
                                "cost_efficiency": cost_per_percent_mde,
                            }
                        )

        # Sort by cost efficiency (lower is better)
        sweet_spots.sort(key=lambda x: x["cost_efficiency"])

        return {
            "oracle_values": oracle_values,
            "sample_values": sample_values,
            "rmse_matrix": rmse_matrix,
            "se_matrix": se_matrix,  # Proper IC-based standard errors
            "mean_rmse_grid": mean_rmse_grid,
            "mean_se_grid": mean_se_grid,
            "success_grid": success_grid,
            "sweet_spots": sweet_spots[:5],  # Top 5 most cost-efficient
        }

    def create_figure(self, results: List[Dict[str, Any]], output_path: Path = None):
        """Create Figure 3: Interaction heatmap with MDE contours."""

        analysis = self.analyze_results(results)

        if not analysis["oracle_values"] or not analysis["sample_values"]:
            logger.warning("No successful results to plot")
            return

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: RMSE heatmap with improved formatting
        ax = axes[0]
        rmse = analysis["rmse_matrix"]
        mask = ~np.isfinite(rmse)
        
        sns.heatmap(
            rmse,
            mask=mask,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",  # Sequential colormap, colorblind-friendly
            vmin=0.0,
            vmax=np.nanquantile(rmse, 0.95),  # Consistent scale across runs
            xticklabels=[str(x) for x in analysis["sample_values"]],
            yticklabels=[f"{int(100*y)}%" for y in analysis["oracle_values"]],
            cbar_kws={"label": "RMSE"},
            ax=ax,
        )
        ax.set_xlabel("Sample Size", fontsize=12)
        ax.set_ylabel("Oracle Coverage", fontsize=12)
        ax.set_title("A. RMSE vs Oracle Truth", fontsize=14, fontweight="bold")

        # Panel B: MDE contours (ability to detect 1-2% differences)
        ax = axes[1]

        # Use proper IC-based standard errors from estimators
        se_grid = np.array(analysis["se_matrix"], dtype=float)

        # Power calculation settings
        alpha = 0.05  # 5% significance level
        power = 0.80  # 80% power
        z_alpha = 1.96  # stats.norm.ppf(1 - alpha/2)
        z_power = 0.84  # stats.norm.ppf(power)
        k = z_alpha + z_power  # ≈ 2.80

        # MDE for single policy and two-policy comparison
        mde_one = k * se_grid
        mde_two = k * np.sqrt(2.0) * se_grid  # Conservative: assumes SE_Δ ≈ √2 * SE

        # Build coordinates in actual units (not indices)
        X, Y = np.meshgrid(analysis["sample_values"], analysis["oracle_values"])

        # Mask invalid cells
        mask = ~np.isfinite(mde_two)
        mde_plot = np.ma.array(mde_two, mask=mask)

        # Filled contours of MDE for two-policy comparison
        cf = ax.contourf(
            X, Y, mde_plot,
            levels=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1],
            cmap="viridis",
            antialiased=True,
        )
        cbar = plt.colorbar(cf, ax=ax, label="MDE (two-policy, 80% power, 95% CI)")

        # Key iso-lines at 1% and 2% effect sizes
        cs = ax.contour(
            X, Y, mde_plot,
            levels=[0.01, 0.02],
            colors=["white", "black"],
            linestyles=["--", "-"],
            linewidths=[2.0, 2.0],
        )
        ax.clabel(cs, fmt={0.01: "1%", 0.02: "2%"}, fontsize=10)

        # Optional: overlay iso-lines of n_oracle (labeling effort)
        n_oracle = X * Y  # X = sample size, Y = oracle coverage fraction
        cost_lines = ax.contour(
            X, Y, np.ma.array(n_oracle, mask=mask),
            levels=[50, 100, 250, 500, 1000, 2000],
            colors="gray",
            linewidths=0.8,
            alpha=0.5,
        )
        ax.clabel(cost_lines, fmt=lambda v: f"{int(v)} labels", fontsize=8)

        ax.set_xlabel("Sample Size", fontsize=12)
        ax.set_ylabel("Oracle Coverage", fontsize=12)
        ax.set_title("B. MDE (Can we detect 1-2% effects?)", fontsize=14, fontweight="bold")
        ax.set_ylim(min(analysis["oracle_values"]), max(analysis["oracle_values"]))
        ax.invert_yaxis()  # Match heatmap orientation (higher coverage at top)

        plt.suptitle(
            "Oracle × Sample Size Interaction", fontsize=16, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle

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
        logger.info("\nMost cost-efficient configurations:")
        logger.info("(For detecting 1-2% improvements with 80% power)")

        for i, spot in enumerate(analysis["sweet_spots"], 1):
            logger.info(
                f"\n{i}. Oracle={spot['oracle_coverage']:.0%}, n={spot['sample_size']}"
            )
            logger.info(f"   N oracle labels: {spot['n_oracle']:.0f}")
            logger.info(f"   RMSE: {spot['rmse']:.4f}")
            logger.info(f"   MDE (two-policy): {spot['mde_two']:.1%}")
            logger.info(f"   Can detect: ≥{spot['achievable_mde']:.0%} effects")
            logger.info(f"   Cost per % MDE: {spot['cost_efficiency']:.0f} labels")

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
    figure_path = Path("results/interaction/interaction_mde_analysis.png")
    ablation.create_figure(results, figure_path)

    logger.info("\n" + "=" * 70)
    logger.info("INTERACTION ABLATION COMPLETE")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
