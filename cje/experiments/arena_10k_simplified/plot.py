#!/usr/bin/env python3
"""Simple plotting for ablation results.

Usage:
    python plot_ablation.py ablation_results.json
    python plot_ablation.py ablation_results.json --output my_plot.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse


# Oracle ground truths
ORACLE_TRUTHS = {
    "clone": 0.7359,
    "parallel_universe_prompt": 0.7553,
    "premium": 0.7399,
    "unhelpful": 0.1440,
}

# Estimator colors and markers
ESTIMATOR_STYLE = {
    "raw-ips": {"color": "#e74c3c", "marker": "o", "label": "Raw IPS"},
    "calibrated-ips": {"color": "#3498db", "marker": "s", "label": "Calibrated IPS"},
    "dr-cpo": {"color": "#2ecc71", "marker": "^", "label": "DR-CPO"},
    "mrdr": {"color": "#f39c12", "marker": "D", "label": "MRDR"},
    "tmle": {"color": "#9b59b6", "marker": "v", "label": "TMLE"},
}


def load_results(path: Path) -> pd.DataFrame:
    """Load and process ablation results."""
    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    for result in data:
        if result.get("success") and result.get("estimates"):
            base_row = {
                "estimator": result["estimator"],
                "oracle_coverage": result["oracle_coverage"],
                "sample_fraction": result.get("sample_fraction", 1.0),
                "n_samples": result.get("n_samples", 994),
                "seed": result.get("seed", 0),
                "ess": result.get("diagnostics", {}).get("ess"),
            }

            for policy, estimate in result["estimates"].items():
                row = base_row.copy()
                row["policy"] = policy
                row["estimate"] = estimate
                rows.append(row)

    return pd.DataFrame(rows)


def plot_by_coverage(df: pd.DataFrame, output: Path = None) -> None:
    """Create plot showing performance vs oracle coverage."""

    # Focus on key policies (exclude unhelpful due to poor overlap)
    policies = ["clone", "parallel_universe_prompt", "premium"]

    fig, axes = plt.subplots(1, len(policies), figsize=(15, 5))

    for idx, policy in enumerate(policies):
        ax = axes[idx]
        df_policy = df[(df["policy"] == policy) & (df["sample_fraction"] == 1.0)]

        for estimator in df_policy["estimator"].unique():
            if estimator not in ESTIMATOR_STYLE:
                continue

            style = ESTIMATOR_STYLE[estimator]
            df_est = df_policy[df_policy["estimator"] == estimator]

            # Group by coverage and compute mean/std
            grouped = df_est.groupby("oracle_coverage")["estimate"].agg(["mean", "std"])

            ax.errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"],
                **style,
                capsize=5,
                alpha=0.8,
                markersize=8,
            )

        # Add oracle truth line
        if policy in ORACLE_TRUTHS:
            ax.axhline(
                ORACLE_TRUTHS[policy],
                color="black",
                linestyle="--",
                alpha=0.5,
                label="Oracle Truth",
            )

        ax.set_xlabel("Oracle Coverage")
        ax.set_ylabel("Policy Value Estimate")
        ax.set_title(policy.replace("_", " ").title())
        ax.set_xscale("log")
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1.0])
        ax.set_xticklabels(["5%", "10%", "20%", "50%", "100%"])
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    plt.suptitle(
        "Ablation Results: Estimator Performance vs Oracle Coverage", fontsize=14
    )
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output}")
    else:
        plt.show()


def plot_summary_table(df: pd.DataFrame, output: Path = None) -> None:
    """Create summary table of results at 100% coverage."""

    # Filter to 100% coverage and full sample
    df_full = df[(df["oracle_coverage"] == 1.0) & (df["sample_fraction"] == 1.0)]

    if df_full.empty:
        print("No results found for 100% oracle coverage")
        return

    # Compute errors vs oracle
    errors = []
    for _, row in df_full.iterrows():
        if row["policy"] in ORACLE_TRUTHS:
            error = abs(row["estimate"] - ORACLE_TRUTHS[row["policy"]])
            errors.append(
                {
                    "estimator": row["estimator"],
                    "policy": row["policy"],
                    "estimate": row["estimate"],
                    "oracle": ORACLE_TRUTHS[row["policy"]],
                    "error": error,
                }
            )

    if not errors:
        print("No oracle truth available for comparison")
        return

    df_errors = pd.DataFrame(errors)

    # Compute mean absolute error by estimator
    mae = df_errors.groupby("estimator")["error"].mean().sort_values()

    print("\n" + "=" * 60)
    print("SUMMARY: Mean Absolute Error at 100% Oracle Coverage")
    print("=" * 60)

    for estimator, error in mae.items():
        label = ESTIMATOR_STYLE.get(estimator, {}).get("label", estimator)
        print(f"{label:20} {error:.4f}")

    print("\nDetailed Results:")
    print("-" * 60)

    for estimator in mae.index:
        df_est = df_errors[df_errors["estimator"] == estimator]
        label = ESTIMATOR_STYLE.get(estimator, {}).get("label", estimator)
        print(f"\n{label}:")

        for _, row in df_est.iterrows():
            print(
                f"  {row['policy']:28} {row['estimate']:.4f} "
                f"(truth: {row['oracle']:.4f}, error: {row['error']:.4f})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation results")
    parser.add_argument("input", help="Path to ablation results JSON")
    parser.add_argument("--output", help="Output plot file")
    parser.add_argument(
        "--table", action="store_true", help="Show summary table instead of plot"
    )

    args = parser.parse_args()

    # Load results
    df = load_results(Path(args.input))
    print(f"Loaded {len(df)} data points from {args.input}")

    if args.table:
        plot_summary_table(df)
    else:
        plot_by_coverage(df, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
