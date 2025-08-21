#!/usr/bin/env python3
"""
Minimal IIC ablation that leverages existing diagnostics.

Since IIC diagnostics are already computed and stored in metadata,
we just need to extract and visualize them properly.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core.base import BaseAblation
from core.schemas import ExperimentSpec

logger = logging.getLogger(__name__)


class IICEffectMinimal(BaseAblation):
    """Minimal IIC ablation using existing diagnostics."""

    def __init__(self) -> None:
        super().__init__("iic_effect_minimal")

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run experiments with IIC enabled (default) to collect diagnostics."""
        results = []

        # IIC is enabled by default in all DR estimators
        # We just need to run them and extract the diagnostics

        for estimator in ["calibrated-ips", "dr-cpo", "tmle"]:
            for oracle_coverage in [0.1, 0.2, 0.5]:
                spec = ExperimentSpec(
                    ablation="iic_effect",
                    dataset_path="../data/cje_dataset.jsonl",
                    estimator=estimator,
                    oracle_coverage=oracle_coverage,
                    n_seeds=5,
                )

                logger.info(f"Running {estimator} (oracle={oracle_coverage*100:.0f}%)")
                seed_results = self.run_with_seeds(spec)

                # Extract IIC diagnostics from metadata
                for result in seed_results:
                    if result.get("success") and "estimates" in result:
                        # Check if IIC diagnostics are in metadata
                        # They should be at result["metadata"]["iic_diagnostics"]
                        result["iic_enabled"] = True
                        result["iic_diagnostics_found"] = False

                        # The diagnostics might be in different places depending on estimator
                        # Let's be flexible in extraction
                        if "iic_diagnostics" in result.get("metadata", {}):
                            result["iic_diagnostics_found"] = True

                results.extend(seed_results)

        return results

    def analyze_results(self, results: List[Dict[str, Any]]) -> None:
        """Extract and analyze IIC diagnostics from results."""

        output_dir = Path("results/iic_effect")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect IIC metrics
        iic_data = []

        for r in results:
            if not r.get("success"):
                continue

            estimator = r["spec"]["estimator"]
            oracle_coverage = r["spec"]["oracle_coverage"]

            # Look for IIC diagnostics in metadata
            if "metadata" in r and isinstance(r["metadata"], dict):
                iic_diag = r["metadata"].get("iic_diagnostics", {})

                for policy, diag in iic_diag.items():
                    if isinstance(diag, dict):
                        iic_data.append(
                            {
                                "estimator": estimator,
                                "policy": policy,
                                "oracle_coverage": oracle_coverage,
                                "r_squared": diag.get("r_squared", 0),
                                "se_reduction": diag.get("se_reduction", 0),
                                "var_reduction": diag.get("var_reduction", 0),
                                "ess_gain": diag.get("ess_gain", 1),
                                "direction": diag.get("direction", "unknown"),
                            }
                        )

        if not iic_data:
            logger.warning("No IIC diagnostics found in results")
            return

        df = pd.DataFrame(iic_data)

        # Generate main plot: SE reduction vs R²
        fig, ax = plt.subplots(figsize=(10, 8))

        for estimator in df["estimator"].unique():
            est_df = df[df["estimator"] == estimator]
            scatter = ax.scatter(
                est_df["r_squared"],
                est_df["se_reduction"] * 100,
                label=estimator,
                alpha=0.7,
                s=50,
            )

        ax.set_xlabel("R² (Judge Explainability)", fontsize=12)
        ax.set_ylabel("SE Reduction (%)", fontsize=12)
        ax.set_title(
            "IIC: Variance Reduction from Judge Score Predictability", fontsize=14
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add summary statistics
        if len(df) > 0:
            mean_r2 = df["r_squared"].mean()
            mean_se_reduction = df["se_reduction"].mean() * 100
            max_se_reduction = df["se_reduction"].max() * 100

            summary_text = (
                f"Mean R²: {mean_r2:.3f}\n"
                f"Mean SE reduction: {mean_se_reduction:.1f}%\n"
                f"Max SE reduction: {max_se_reduction:.1f}%"
            )
            ax.text(
                0.02,
                0.98,
                summary_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat"),
            )

        plt.tight_layout()
        plt.savefig(output_dir / "iic_se_reduction_vs_r2.png", dpi=150)
        plt.close()

        # Summary table
        summary = (
            df.groupby(["estimator", "policy"])
            .agg(
                {
                    "r_squared": "mean",
                    "se_reduction": "mean",
                    "var_reduction": "mean",
                    "ess_gain": "mean",
                }
            )
            .round(3)
        )

        print("\n" + "=" * 60)
        print("IIC SUMMARY")
        print("=" * 60)
        print(summary)

        # Save summary
        summary.to_csv(output_dir / "iic_summary.csv")
        logger.info(f"Results saved to {output_dir}")


def main() -> None:
    """Run minimal IIC ablation."""
    ablation = IICEffectMinimal()
    results = ablation.run_ablation()
    ablation.analyze_results(results)


if __name__ == "__main__":
    main()
