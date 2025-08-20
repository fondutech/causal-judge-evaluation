#!/usr/bin/env python3
"""Sample size scaling ablation.

This ablation answers: How many samples do we need?

Key findings we expect:
- Standard errors follow √n scaling
- DR methods are ~4x more sample efficient than IPS
- ESS degrades with importance weighting
- Minimum viable is ~1000 samples for production use
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


class SampleSizeAblation(BaseAblation):
    """Ablation to study sample size requirements."""

    def __init__(self):
        super().__init__(
            name="sample_size", cache_dir=Path("../.ablation_cache/sample_size")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run sample size scaling experiments."""

        # Define sample sizes to test
        sample_sizes = [100, 250, 500, 1000, 2500, 5000]

        # We'll test two estimators to show efficiency difference
        estimators = ["calibrated-ips", "mrdr"]

        # Fixed parameters
        oracle_coverage = 0.10  # 10% oracle (reasonable default)
        n_seeds = 5  # Multiple seeds for stability

        logger.info("=" * 70)
        logger.info("SAMPLE SIZE SCALING ABLATION")
        logger.info("=" * 70)
        logger.info(f"Sample sizes: {sample_sizes}")
        logger.info(f"Estimators: {estimators}")
        logger.info(f"Oracle coverage: {oracle_coverage:.0%}")
        logger.info(f"Seeds per configuration: {n_seeds}")
        logger.info("")

        all_results = []

        for estimator in estimators:
            logger.info(f"\n{'='*60}")
            logger.info(f"Estimator: {estimator.upper()}")
            logger.info(f"{'='*60}")

            for n_samples in sample_sizes:
                logger.info(f"\nSample size: {n_samples}")
                logger.info("-" * 30)

                # Determine correct data path based on current directory
                data_path = Path("../data/cje_dataset.jsonl")
                if not data_path.exists():
                    data_path = Path("../../data/cje_dataset.jsonl")

                spec = ExperimentSpec(
                    ablation="sample_size",
                    dataset_path=str(data_path),
                    estimator=estimator,
                    oracle_coverage=oracle_coverage,
                    sample_size=n_samples,
                    n_seeds=n_seeds,
                    seed_base=42,
                )

                # Run with multiple seeds
                results = self.run_with_seeds(spec)
                all_results.extend(results)

                # Show aggregate statistics
                agg = aggregate_results(results)
                if agg.get("n_seeds_successful", 0) > 0:
                    # Extract successful results
                    successful_results = [r for r in results if r.get("success", False)]

                    # Compute mean metrics
                    mean_rmse = np.mean(
                        [r.get("rmse_vs_oracle", np.nan) for r in successful_results]
                    )

                    # Mean SE across policies and seeds
                    all_ses = []
                    for r in successful_results:
                        if "standard_errors" in r:
                            all_ses.extend(list(r["standard_errors"].values()))
                    mean_se = np.mean(all_ses) if all_ses else np.nan

                    # Mean ESS
                    all_ess = []
                    for r in successful_results:
                        if "ess_absolute" in r:
                            all_ess.extend(list(r["ess_absolute"].values()))
                    mean_ess = np.mean(all_ess) if all_ess else np.nan

                    logger.info(
                        f"Results ({agg['n_seeds_successful']}/{agg['n_seeds_total']} successful):"
                    )
                    logger.info(f"  Mean RMSE: {mean_rmse:.4f}")
                    logger.info(f"  Mean SE: {mean_se:.4f}")
                    logger.info(
                        f"  Mean ESS: {mean_ess:.0f} ({100*mean_ess/n_samples:.1f}%)"
                    )
                else:
                    logger.warning(f"All {agg['n_seeds_total']} seeds failed!")

        # Save all results
        output_dir = Path("../ablations/results/sample_size")
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
        """Analyze sample size scaling."""

        # Group by estimator and sample size
        by_config = {}
        for r in results:
            if r.get("success", False):
                estimator = r["spec"]["estimator"]
                n_samples = r["spec"]["sample_size"]
                key = (estimator, n_samples)
                if key not in by_config:
                    by_config[key] = []
                by_config[key].append(r)

        # Organize analysis by estimator
        estimators = list(set(k[0] for k in by_config.keys()))
        analysis = {
            est: {
                "sample_sizes": [],
                "mean_rmse": [],
                "mean_se": [],
                "mean_ess": [],
                "ess_percent": [],
            }
            for est in estimators
        }

        for (estimator, n_samples), results_subset in sorted(by_config.items()):
            # RMSE
            rmses = [r.get("rmse_vs_oracle", np.nan) for r in results_subset]

            # Standard errors
            all_ses = []
            for r in results_subset:
                if "standard_errors" in r:
                    all_ses.extend(list(r["standard_errors"].values()))

            # ESS
            all_ess = []
            for r in results_subset:
                if "ess_absolute" in r:
                    all_ess.extend(list(r["ess_absolute"].values()))

            analysis[estimator]["sample_sizes"].append(n_samples)
            analysis[estimator]["mean_rmse"].append(np.nanmean(rmses))
            analysis[estimator]["mean_se"].append(
                np.nanmean(all_ses) if all_ses else np.nan
            )
            analysis[estimator]["mean_ess"].append(
                np.nanmean(all_ess) if all_ess else np.nan
            )
            analysis[estimator]["ess_percent"].append(
                100 * np.nanmean(all_ess) / n_samples if all_ess else np.nan
            )

        # Check √n scaling
        for estimator in estimators:
            n_array = np.array(analysis[estimator]["sample_sizes"])
            se_array = np.array(analysis[estimator]["mean_se"])

            # Filter out NaNs
            mask = np.isfinite(se_array) & (n_array > 0)
            if np.sum(mask) >= 2:
                # Fit SE = c / √n in log space
                log_n = np.log(n_array[mask])
                log_se = np.log(se_array[mask])

                # Linear regression in log space
                from scipy import stats

                slope, intercept, r_value, _, _ = stats.linregress(log_n, log_se)

                analysis[estimator]["scaling_exponent"] = slope
                analysis[estimator]["scaling_r2"] = r_value**2

                # Ideal √n scaling has slope = -0.5
                analysis[estimator]["follows_sqrt_n"] = abs(slope + 0.5) < 0.1

        return analysis

    def create_figure(self, results: List[Dict[str, Any]], output_path: Path = None):
        """Create Figure 2: Sample size scaling."""

        analysis = self.analyze_results(results)

        if not analysis:
            logger.warning("No successful results to plot")
            return

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        colors = {"calibrated-ips": "blue", "mrdr": "orange"}

        # Panel A: RMSE vs n
        ax = axes[0, 0]
        for estimator, data in analysis.items():
            if data["sample_sizes"]:
                ax.loglog(
                    data["sample_sizes"],
                    data["mean_rmse"],
                    "o-",
                    label=estimator.upper(),
                    color=colors.get(estimator, "gray"),
                    linewidth=2,
                    markersize=8,
                )
        ax.set_xlabel("Sample Size (n)", fontsize=12)
        ax.set_ylabel("RMSE vs Oracle", fontsize=12)
        ax.set_title(
            "A. Estimation Error vs Sample Size", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel B: SE vs n (with √n reference)
        ax = axes[0, 1]
        for estimator, data in analysis.items():
            if data["sample_sizes"]:
                ax.loglog(
                    data["sample_sizes"],
                    data["mean_se"],
                    "o-",
                    label=estimator.upper(),
                    color=colors.get(estimator, "gray"),
                    linewidth=2,
                    markersize=8,
                )

        # Add √n reference line
        n_ref = np.array([100, 5000])
        se_ref = 0.1 * np.sqrt(n_ref[0] / n_ref)  # c/√n scaling
        ax.loglog(n_ref, se_ref, "k--", alpha=0.5, label="√n scaling")

        ax.set_xlabel("Sample Size (n)", fontsize=12)
        ax.set_ylabel("Standard Error", fontsize=12)
        ax.set_title("B. Standard Error Scaling", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add scaling exponent annotation
        for i, (estimator, data) in enumerate(analysis.items()):
            if "scaling_exponent" in data:
                ax.text(
                    0.05,
                    0.95 - i * 0.1,
                    f'{estimator}: slope = {data["scaling_exponent"]:.2f}',
                    transform=ax.transAxes,
                    fontsize=10,
                )

        # Panel C: ESS vs n
        ax = axes[1, 0]
        for estimator, data in analysis.items():
            if data["sample_sizes"]:
                ax.plot(
                    data["sample_sizes"],
                    data["mean_ess"],
                    "o-",
                    label=estimator.upper(),
                    color=colors.get(estimator, "gray"),
                    linewidth=2,
                    markersize=8,
                )

        # Add ideal line (ESS = n)
        n_ideal = np.array([100, 5000])
        ax.plot(n_ideal, n_ideal, "k--", alpha=0.5, label="Ideal (ESS=n)")

        ax.set_xlabel("Sample Size (n)", fontsize=12)
        ax.set_ylabel("Effective Sample Size", fontsize=12)
        ax.set_title("C. Effective Sample Size", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel D: ESS percentage
        ax = axes[1, 1]
        for estimator, data in analysis.items():
            if data["sample_sizes"]:
                ax.semilogx(
                    data["sample_sizes"],
                    data["ess_percent"],
                    "o-",
                    label=estimator.upper(),
                    color=colors.get(estimator, "gray"),
                    linewidth=2,
                    markersize=8,
                )

        ax.axhline(100, color="k", linestyle="--", alpha=0.5, label="Ideal (100%)")
        ax.axhline(
            10, color="r", linestyle="--", alpha=0.5, label="Gate threshold (10%)"
        )

        ax.set_xlabel("Sample Size (n)", fontsize=12)
        ax.set_ylabel("ESS / n (%)", fontsize=12)
        ax.set_title("D. Relative Efficiency", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 110])

        plt.suptitle(
            "Sample Size Requirements for CJE", fontsize=16, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure to {output_path}")

        plt.show()

        return fig


def main():
    """Run sample size ablation."""

    ablation = SampleSizeAblation()

    # Run the ablation
    results = ablation.run_ablation()

    # Analyze
    analysis = ablation.analyze_results(results)

    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)

    for estimator, data in analysis.items():
        logger.info(f"\n{estimator.upper()}:")

        if data["sample_sizes"]:
            logger.info("\n  Sample Size -> RMSE:")
            for n, rmse in zip(data["sample_sizes"], data["mean_rmse"]):
                logger.info(f"    {n:5d}: {rmse:.4f}")

            if "scaling_exponent" in data:
                logger.info(f"\n  SE scaling exponent: {data['scaling_exponent']:.3f}")
                logger.info(f"  (Ideal √n scaling = -0.50)")
                logger.info(f"  R² of fit: {data['scaling_r2']:.3f}")

                if data["follows_sqrt_n"]:
                    logger.info("  ✓ Follows √n scaling")
                else:
                    logger.info("  ✗ Deviates from √n scaling")

    # Compare efficiency
    if len(analysis) == 2 and all(
        len(data["mean_rmse"]) > 0 for data in analysis.values()
    ):
        estimators = list(analysis.keys())

        # Find common sample sizes
        common_sizes = set(analysis[estimators[0]]["sample_sizes"]) & set(
            analysis[estimators[1]]["sample_sizes"]
        )

        if common_sizes:
            logger.info(f"\nEfficiency comparison at n={max(common_sizes)}:")
            n = max(common_sizes)

            for est in estimators:
                idx = analysis[est]["sample_sizes"].index(n)
                rmse = analysis[est]["mean_rmse"][idx]
                logger.info(f"  {est}: RMSE = {rmse:.4f}")

    # Create figure
    figure_path = Path("../ablations/results/sample_size/figure_2_sample_scaling.png")
    ablation.create_figure(results, figure_path)

    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE SIZE ABLATION COMPLETE")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
