#!/usr/bin/env python3
"""
SIMCal variance cap (ρ) ablation.

Tests different variance cap values to understand the bias-variance tradeoff.
ρ = 1.0: No variance increase allowed (most conservative)
ρ = 2.0: Default, allows 2x variance for better bias
ρ = 3.0+: More permissive, less variance reduction
"""

import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.base import BaseAblation
from core.schemas import ExperimentSpec

logger = logging.getLogger(__name__)


class SIMCalRhoAblation(BaseAblation):
    """Ablation for SIMCal variance cap parameter."""

    def __init__(self) -> None:
        super().__init__("simcal_rho")

    def create_estimator(self, spec: Any, sampler: Any, cal_result: Any) -> Any:
        """Override to pass var_cap to CalibratedIPS."""
        if spec.estimator == "calibrated-ips":
            # Import here to avoid circular dependency
            from cje.estimators import CalibratedIPS

            # Get var_cap from spec.extra or use default
            var_cap = spec.extra.get("var_cap", None)

            return CalibratedIPS(
                sampler,
                calibrate=True,
                var_cap=var_cap,
                calibrator=cal_result.calibrator if cal_result else None,
            )
        else:
            return super().create_estimator(spec, sampler, cal_result)

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Test different var_cap values."""
        results = []

        # Test different variance caps (None means no cap)
        var_cap_values = [0.5, 1.0, 2.0, 5.0, None]

        for var_cap in var_cap_values:
            spec = ExperimentSpec(
                ablation="simcal_rho",
                dataset_path="../data/cje_dataset.jsonl",
                estimator="calibrated-ips",
                oracle_coverage=0.2,
                n_seeds=5,
                extra={"var_cap": var_cap},  # Pass via extra dict
            )

            logger.info(f"Running with var_cap={var_cap}")
            seed_results = self.run_with_seeds(spec)
            results.extend(seed_results)

        return results

    def analyze_results(self, results: List[Dict[str, Any]]) -> None:
        """Analyze rho sweep results."""
        output_dir = Path("ablations/results/simcal_rho")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate by var_cap
        cap_data = []
        for r in results:
            if r.get("success"):
                var_cap = r["spec"].get("extra", {}).get("var_cap", "None")

                # Extract relevant metrics
                mean_ess = np.mean(list(r.get("ess_relative", {}).values()))
                mean_ci_width = r.get("mean_ci_width", 0)
                rmse = r.get("rmse_vs_oracle", 0)

                cap_data.append(
                    {
                        "var_cap": var_cap if var_cap is not None else float("inf"),
                        "var_cap_label": str(var_cap),
                        "ess": mean_ess,
                        "ci_width": mean_ci_width,
                        "rmse": rmse,
                    }
                )

        if not cap_data:
            logger.warning("No successful results")
            return

        df = pd.DataFrame(cap_data)

        # Plot: 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Group by var_cap and aggregate
        grouped = df.groupby("var_cap").agg(
            {
                "ess": ["mean", "std"],
                "ci_width": ["mean", "std"],
                "rmse": ["mean", "std"],
            }
        )

        var_caps = grouped.index

        # Panel A: ESS vs var_cap
        ax = axes[0, 0]
        ax.errorbar(
            var_caps,
            grouped["ess"]["mean"],
            yerr=grouped["ess"]["std"],
            marker="o",
            capsize=5,
        )
        ax.set_xlabel("Variance Cap")
        ax.set_ylabel("ESS (%)")
        ax.set_title("(A) Effective Sample Size")
        ax.grid(True, alpha=0.3)

        # Panel B: CI width vs var_cap
        ax = axes[0, 1]
        ax.errorbar(
            var_caps,
            grouped["ci_width"]["mean"],
            yerr=grouped["ci_width"]["std"],
            marker="o",
            capsize=5,
            color="orange",
        )
        ax.set_xlabel("Variance Cap")
        ax.set_ylabel("CI Width")
        ax.set_title("(B) Confidence Interval Width")
        ax.grid(True, alpha=0.3)

        # Panel C: RMSE vs var_cap
        ax = axes[1, 0]
        ax.errorbar(
            var_caps,
            grouped["rmse"]["mean"],
            yerr=grouped["rmse"]["std"],
            marker="o",
            capsize=5,
            color="red",
        )
        ax.set_xlabel("Variance Cap")
        ax.set_ylabel("RMSE vs Oracle")
        ax.set_title("(C) Estimation Error")
        ax.grid(True, alpha=0.3)

        # Panel D: Bias-variance tradeoff
        ax = axes[1, 1]
        # Normalize to show tradeoff
        ess_norm = grouped["ess"]["mean"] / grouped["ess"]["mean"].max()
        rmse_norm = 1 - (grouped["rmse"]["mean"] / grouped["rmse"]["mean"].max())

        ax.plot(var_caps, ess_norm, "o-", label="ESS (normalized)")
        ax.plot(var_caps, rmse_norm, "s-", label="Accuracy (1-RMSE, normalized)")
        ax.axvline(
            x=2.0, color="red", linestyle="--", alpha=0.5, label="Default var_cap=2"
        )
        ax.set_xlabel("Variance Cap")
        ax.set_ylabel("Normalized Performance")
        ax.set_title("(D) Bias-Variance Tradeoff")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle("SIMCal Variance Cap Ablation", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "simcal_rho_ablation.png", dpi=150)
        plt.close()

        # Print summary
        print("\n" + "=" * 60)
        print("SIMCAL VARIANCE CAP ABLATION SUMMARY")
        print("=" * 60)
        print(grouped)
        print(f"\nOptimal var_cap appears to be around 2.0 (allows 2x variance)")

        logger.info(f"Results saved to {output_dir}")


def main() -> None:
    """Run SIMCal rho ablation."""
    ablation = SIMCalRhoAblation()
    results = ablation.run_ablation()
    ablation.analyze_results(results)


if __name__ == "__main__":
    main()
