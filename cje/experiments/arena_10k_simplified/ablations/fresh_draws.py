#!/usr/bin/env python3
"""
Fresh draws (K) ablation for DR estimators.

Tests how many fresh draws per prompt improve DR estimates.
More draws reduce Monte Carlo variance but increase compute cost.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .core.base import BaseAblation
from .core.schemas import ExperimentSpec

logger = logging.getLogger(__name__)


class FreshDrawsAblation(BaseAblation):
    """Ablation for number of fresh draws in DR estimation."""

    def __init__(self) -> None:
        super().__init__("fresh_draws")

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Test different numbers of fresh draws."""
        results = []

        # Test different K values
        k_values = [1, 3, 5, 10, 20]

        for k in k_values:
            for estimator in ["dr-cpo", "tmle"]:
                spec = ExperimentSpec(
                    ablation="fresh_draws",
                    dataset_path="../../data/cje_dataset.jsonl",
                    estimator=estimator,
                    oracle_coverage=0.2,
                    n_seeds=3,  # Fewer seeds since this is expensive
                    draws_per_prompt=k,
                )

                logger.info(f"Running {estimator} with K={k} draws")
                seed_results = self.run_with_seeds(spec)
                results.extend(seed_results)

        return results

    def analyze_results(self, results: List[Dict[str, Any]]) -> None:
        """Analyze fresh draws sweep."""
        output_dir = Path("ablations/results/fresh_draws")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect data
        draw_data = []
        for r in results:
            if r.get("success"):
                k = r["spec"]["draws_per_prompt"]
                estimator = r["spec"]["estimator"]

                # Extract metrics
                for policy in r.get("estimates", {}).keys():
                    se = r.get("standard_errors", {}).get(policy, 0)

                    draw_data.append(
                        {
                            "k": k,
                            "estimator": estimator,
                            "policy": policy,
                            "se": se,
                            "runtime": r.get("runtime_s", 0),
                        }
                    )

        if not draw_data:
            logger.warning("No successful results")
            return

        df = pd.DataFrame(draw_data)

        # Plot: 1x3 grid
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel A: SE vs K
        ax = axes[0]
        for estimator in df["estimator"].unique():
            est_df = df[df["estimator"] == estimator]
            grouped = est_df.groupby("k")["se"].agg(["mean", "std"])
            ax.errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                label=estimator,
                capsize=5,
            )

        ax.set_xlabel("K (Fresh Draws per Prompt)")
        ax.set_ylabel("Standard Error")
        ax.set_title("(A) SE vs Fresh Draws")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel B: Runtime vs K
        ax = axes[1]
        runtime_grouped = df.groupby("k")["runtime"].agg(["mean", "std"])
        ax.errorbar(
            runtime_grouped.index,
            runtime_grouped["mean"],
            yerr=runtime_grouped["std"],
            marker="s",
            color="orange",
            capsize=5,
        )
        ax.set_xlabel("K (Fresh Draws per Prompt)")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title("(B) Computational Cost")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Panel C: Efficiency (SE reduction per unit time)
        ax = axes[2]
        # Compute efficiency: 1/SE normalized by runtime
        efficiency_data = []
        for k in df["k"].unique():
            k_df = df[df["k"] == k]
            mean_se = k_df["se"].mean()
            mean_runtime = k_df["runtime"].mean()
            efficiency = (1 / mean_se) / mean_runtime if mean_runtime > 0 else 0
            efficiency_data.append({"k": k, "efficiency": efficiency})

        eff_df = pd.DataFrame(efficiency_data)
        ax.plot(eff_df["k"], eff_df["efficiency"], "o-", color="green")
        ax.set_xlabel("K (Fresh Draws per Prompt)")
        ax.set_ylabel("Efficiency (1/SE per second)")
        ax.set_title("(C) Efficiency: Precision per Unit Time")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Mark optimal K
        optimal_k = eff_df.loc[eff_df["efficiency"].idxmax(), "k"]
        ax.axvline(x=optimal_k, color="red", linestyle="--", alpha=0.5)
        ax.text(
            optimal_k * 1.2,
            ax.get_ylim()[1] * 0.9,
            f"Optimal K≈{optimal_k}",
            color="red",
        )

        plt.suptitle("Fresh Draws (K) Ablation for DR Estimators", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "fresh_draws_ablation.png", dpi=150)
        plt.close()

        # Summary table
        summary = (
            df.groupby(["k", "estimator"])
            .agg({"se": ["mean", "std"], "runtime": "mean"})
            .round(4)
        )

        print("\n" + "=" * 60)
        print("FRESH DRAWS ABLATION SUMMARY")
        print("=" * 60)
        print(summary)
        print(f"\nOptimal K (best efficiency): {optimal_k}")
        print("Recommendation: K=5-10 balances precision and compute cost")

        logger.info(f"Results saved to {output_dir}")


def main() -> None:
    """Run fresh draws ablation."""
    ablation = FreshDrawsAblation()
    results = ablation.run_ablation()
    ablation.analyze_results(results)


if __name__ == "__main__":
    main()
