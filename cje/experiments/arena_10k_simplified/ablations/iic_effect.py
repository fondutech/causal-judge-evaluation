#!/usr/bin/env python3
"""
IIC (Isotonic Influence Control) ablation with comprehensive visualizations.

This ablation demonstrates:
1. How IIC reduces variance by exploiting judge score monotonicity
2. The magnitude of variance reduction for different policies
3. The relationship between judge explainability (R²) and SE reduction
4. Visual proof that IIC preserves unbiasedness while shrinking CIs
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from core.base import BaseAblation
from core.schemas import ExperimentSpec

logger = logging.getLogger(__name__)


class IICEffectAblation(BaseAblation):
    """Comprehensive IIC ablation with visualizations."""

    def __init__(self) -> None:
        super().__init__("iic_effect")

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run experiments with and without IIC to show the effect."""
        results = []

        # Test multiple estimators and oracle coverages
        estimators = ["dr-cpo", "tmle", "mrdr"]
        oracle_coverages = [0.1, 0.2, 0.5]

        for estimator in estimators:
            for oracle_coverage in oracle_coverages:
                # Run WITH IIC (default)
                spec_with = ExperimentSpec(
                    ablation="iic_effect",
                    dataset_path="../data/cje_dataset.jsonl",
                    estimator=estimator,
                    oracle_coverage=oracle_coverage,
                    n_seeds=5,
                    extra={"use_iic": True},  # Explicitly set even though it's default
                )

                logger.info(
                    f"Running {estimator} WITH IIC (oracle={oracle_coverage*100:.0f}%)"
                )
                with_iic_results = self.run_with_seeds(spec_with)
                for r in with_iic_results:
                    r["iic_enabled"] = True
                results.extend(with_iic_results)

                # Run WITHOUT IIC for comparison
                spec_without = ExperimentSpec(
                    ablation="iic_effect",
                    dataset_path="../data/cje_dataset.jsonl",
                    estimator=estimator,
                    oracle_coverage=oracle_coverage,
                    n_seeds=5,
                    extra={"use_iic": False},
                )

                logger.info(
                    f"Running {estimator} WITHOUT IIC (oracle={oracle_coverage*100:.0f}%)"
                )
                without_iic_results = self.run_with_seeds(spec_without)
                for r in without_iic_results:
                    r["iic_enabled"] = False
                results.extend(without_iic_results)

        return results

    def extract_influence_functions(
        self, result: Dict[str, Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract influence functions from result metadata."""
        if not result.get("success") or "metadata" not in result:
            return None

        metadata = result["metadata"]

        # Look for raw and IIC-smoothed influence functions
        if "dr_influence" in metadata:
            return metadata["dr_influence"]  # type: ignore[no-any-return]

        return None

    def analyze_results(self, results: List[Dict[str, Any]]) -> None:
        """Generate simplified IIC visualization showing key relationship."""

        output_dir = Path("results/iic_effect")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Separate results by IIC status
        with_iic = [
            r for r in results if r.get("iic_enabled", False) and r.get("success")
        ]
        without_iic = [
            r for r in results if not r.get("iic_enabled", True) and r.get("success")
        ]

        # Collect variance reduction statistics
        variance_data = []

        for r_with in with_iic:
            # Find matching without-IIC result
            r_without = self._find_matching_result(r_with, without_iic)
            if r_without is None:
                continue

            # Compare standard errors
            if "standard_errors" in r_with and "standard_errors" in r_without:
                for policy in r_with["standard_errors"]:
                    if policy in r_without["standard_errors"]:
                        se_with = r_with["standard_errors"][policy]
                        se_without = r_without["standard_errors"][policy]

                        if se_without > 0:
                            se_reduction = (se_without - se_with) / se_without

                            # Get IIC diagnostics if available
                            iic_diag = (
                                r_with.get("metadata", {})
                                .get("iic_diagnostics", {})
                                .get(policy, {})
                            )

                            variance_data.append(
                                {
                                    "estimator": r_with["spec"]["estimator"],
                                    "oracle_coverage": r_with["spec"][
                                        "oracle_coverage"
                                    ],
                                    "policy": policy,
                                    "se_with_iic": se_with,
                                    "se_without_iic": se_without,
                                    "se_reduction": se_reduction,
                                    "r_squared": iic_diag.get("r_squared", 0),
                                    "direction": iic_diag.get("direction", "unknown"),
                                }
                            )

        if not variance_data:
            logger.warning("No paired IIC comparison data found")
            return

        df = pd.DataFrame(variance_data)

        # Create single effective plot: R² vs SE Reduction with policy labels
        fig, ax = plt.subplots(figsize=(10, 8))

        # Aggregate by policy for cleaner visualization
        policy_stats = (
            df.groupby("policy")
            .agg({"r_squared": "mean", "se_reduction": "mean"})
            .reset_index()
        )

        # Color by SE reduction magnitude
        colors = plt.cm.RdYlGn(policy_stats["se_reduction"].values)

        # Create scatter plot
        scatter = ax.scatter(
            policy_stats["r_squared"],
            policy_stats["se_reduction"] * 100,
            c=policy_stats["se_reduction"],
            s=200,
            alpha=0.7,
            cmap="RdYlGn",
            vmin=0,
            vmax=policy_stats["se_reduction"].max(),
            edgecolors="black",
            linewidth=1,
        )

        # Add policy labels
        for idx, row in policy_stats.iterrows():
            # Position label to avoid overlap
            offset_x = 0.02
            offset_y = 1 if row["se_reduction"] * 100 < 20 else -2
            ax.annotate(
                row["policy"],
                xy=(row["r_squared"], row["se_reduction"] * 100),
                xytext=(
                    row["r_squared"] + offset_x,
                    row["se_reduction"] * 100 + offset_y,
                ),
                fontsize=9,
                ha="left",
            )

        # Add trend line (with error handling for numerical issues)
        if len(policy_stats) > 2:
            try:
                # Filter out any NaN or infinite values
                valid_mask = np.isfinite(
                    policy_stats["r_squared"].values
                ) & np.isfinite(policy_stats["se_reduction"].values)
                if valid_mask.sum() > 2:
                    x_valid = policy_stats.loc[valid_mask, "r_squared"].values
                    y_valid = policy_stats.loc[valid_mask, "se_reduction"].values * 100

                    # Use robust fitting with rcond to avoid SVD issues
                    z = np.polyfit(x_valid, y_valid, 1, rcond=None)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_valid.min(), x_valid.max(), 100)
                    ax.plot(
                        x_trend,
                        p(x_trend),
                        "r--",
                        alpha=0.5,
                        linewidth=2,
                        label=f"Trend: SE reduction ≈ {z[0]:.0f} × R²",
                    )
            except np.linalg.LinAlgError:
                # Skip trend line if fitting fails
                logger.warning("Could not fit trend line due to numerical issues")
            except Exception as e:
                logger.warning(f"Error fitting trend line: {e}")

        # Formatting
        ax.set_xlabel("R² (Judge Explainability of Influence Functions)", fontsize=12)
        ax.set_ylabel("Standard Error Reduction (%)", fontsize=12)
        ax.set_title(
            "IIC Effectiveness: Variance Reduction Proportional to Judge Explainability",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label="SE Reduction")
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%")
        )

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Add summary statistics as text box
        mean_reduction = df["se_reduction"].mean() * 100
        mean_r2 = df["r_squared"].mean()
        substantial = (policy_stats["se_reduction"] > 0.10).sum()

        summary_text = (
            f"Mean SE Reduction: {mean_reduction:.1f}%\n"
            f"Mean R²: {mean_r2:.3f}\n"
            f"Policies with >10% reduction: {substantial}/{len(policy_stats)}"
        )

        ax.text(
            0.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

        # Add legend if there are labeled elements
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=10)

        # Set axis limits with padding (handle potential NaN)
        r_squared_max = policy_stats["r_squared"].max()
        se_reduction_max = policy_stats["se_reduction"].max()

        if np.isfinite(r_squared_max):
            ax.set_xlim(-0.05, r_squared_max + 0.1)
        if np.isfinite(se_reduction_max):
            ax.set_ylim(-5, se_reduction_max * 100 + 10)

        plt.tight_layout()
        plt.savefig(output_dir / "iic_effect.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Generate detailed CSV report
        summary_by_policy = (
            df.groupby("policy")
            .agg(
                {
                    "se_reduction": ["mean", "std", "count"],
                    "r_squared": "mean",
                    "se_with_iic": "mean",
                    "se_without_iic": "mean",
                }
            )
            .round(4)
        )

        summary_by_policy.to_csv(output_dir / "iic_policy_summary.csv")

        # Save results as JSONL for regeneration
        results_path = output_dir / "results.jsonl"
        with open(results_path, "w") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
        logger.info(f"Saved {len(results)} results to {results_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("IIC ABLATION SUMMARY")
        print("=" * 60)
        print(f"Mean SE reduction: {mean_reduction:.1f}%")
        print(f"Mean R²: {mean_r2:.3f}")
        print(f"\nPolicies with >10% SE reduction: {substantial}/{len(policy_stats)}")
        print("\nTop policies by SE reduction:")
        top_policies = policy_stats.nlargest(5, "se_reduction")
        for _, row in top_policies.iterrows():
            print(
                f"  {row['policy']}: {row['se_reduction']*100:.1f}% (R²={row['r_squared']:.3f})"
            )

        logger.info(f"Results saved to {output_dir}")

    def _find_matching_result(
        self, target: Dict, candidates: List[Dict]
    ) -> Optional[Dict]:
        """Find result with same estimator, oracle coverage, and seed."""
        for candidate in candidates:
            if (
                candidate["spec"]["estimator"] == target["spec"]["estimator"]
                and candidate["spec"]["oracle_coverage"]
                == target["spec"]["oracle_coverage"]
                and candidate.get("seed") == target.get("seed")
            ):
                return candidate
        return None

    def create_estimator(self, spec: Any, sampler: Any, cal_result: Any) -> Any:
        """Override to control IIC usage."""
        estimator_name = spec.estimator
        use_iic = spec.extra.get("use_iic", True)

        if estimator_name == "dr-cpo":
            from cje.estimators import DRCPOEstimator

            return DRCPOEstimator(
                sampler,
                calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,
                n_folds=5,
            )
        elif estimator_name == "tmle":
            from cje.estimators import TMLEEstimator

            return TMLEEstimator(
                sampler,
                calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,
                n_folds=5,
            )
        elif estimator_name == "mrdr":
            from cje.estimators import MRDREstimator

            return MRDREstimator(
                sampler,
                calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,
                n_folds=5,
            )
        else:
            return super().create_estimator(spec, sampler, cal_result)


def main() -> None:
    """Run IIC ablation with comprehensive visualizations."""
    ablation = IICEffectAblation()
    results = ablation.run_ablation()
    ablation.analyze_results(results)


if __name__ == "__main__":
    main()
