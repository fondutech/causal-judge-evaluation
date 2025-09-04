#!/usr/bin/env python3
"""
Estimator comparison visualization adapted from legacy code.

Creates multi-panel plots comparing estimator performance across different scenarios.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_results(path: str = "results/all_experiments.jsonl") -> List[Dict]:
    """Load experiment results from unified ablation output."""
    results = []
    with open(path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("success"):
                    results.append(data)
            except:
                pass
    return results


def create_estimator_labels(
    estimator: str, use_cal: bool, use_iic: bool, weight_mode: str
) -> str:
    """Create descriptive label for estimator configuration."""
    if estimator == "ips":
        if use_cal:
            if weight_mode == "hajek":
                return "Cal-IPS"
            else:
                return "Cal-IPS (raw)"
        else:
            if weight_mode == "hajek":
                return "SNIPS"
            else:
                return "Raw IPS"

    elif estimator == "dr-cpo":
        base = "DR-CPO"
        if use_cal and use_iic:
            return f"Cal-{base}+IIC"
        elif use_cal:
            return f"Cal-{base}"
        elif use_iic:
            return f"{base}+IIC"
        else:
            return base

    elif estimator == "stacked-dr":
        base = "Stacked-DR"
        if use_cal and use_iic:
            return f"Cal-{base}+IIC"
        elif use_cal:
            return f"Cal-{base}"
        elif use_iic:
            return f"{base}+IIC"
        else:
            return base

    return estimator.upper()


def analyze_by_scenario(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group and analyze results by scenario (sample size × oracle coverage)."""

    scenarios = defaultdict(list)

    for r in results:
        spec = r["spec"]
        extra = spec.get("extra", {})

        # Create scenario key
        n = spec["sample_size"]
        oracle = spec["oracle_coverage"]
        scenario = f"n={n}, oracle={int(oracle*100)}%"

        # Get estimator configuration
        estimator = spec["estimator"]
        use_cal = extra.get("use_calibration", False)
        use_iic = extra.get("use_iic", False)
        weight_mode = extra.get("weight_mode", "hajek")

        # Create label
        label = create_estimator_labels(estimator, use_cal, use_iic, weight_mode)

        # Store result
        scenarios[scenario].append(
            {
                "label": label,
                "estimator": estimator,
                "use_cal": use_cal,
                "use_iic": use_iic,
                "weight_mode": weight_mode,
                "rmse": r.get("rmse_vs_oracle", np.nan),
                "mean_se": (
                    np.mean(list(r.get("standard_errors", {}).values()))
                    if r.get("standard_errors")
                    else np.nan
                ),
            }
        )

    return dict(scenarios)


def compute_rankings(
    scenarios: Dict[str, List[Dict]], metric: str = "se"
) -> Dict[str, List[Dict]]:
    """Compute mean standard error and rank estimators for each scenario.

    Args:
        scenarios: Dictionary of scenario results
        metric: "se" for standard error or "rmse" for RMSE
    """

    rankings = {}

    for scenario, results in scenarios.items():
        # Group by estimator label
        by_estimator = defaultdict(list)
        for r in results:
            if metric == "se":
                value = r.get("mean_se", np.nan)
            else:
                value = r.get("rmse", np.nan)

            if not np.isnan(value):
                by_estimator[r["label"]].append(value)

        # Compute means and sort
        estimator_means = []
        for label, values in by_estimator.items():
            estimator_means.append(
                {
                    "estimator": label,
                    "mean_value": np.mean(values),
                    "std_value": np.std(values),
                    "n_runs": len(values),
                    "mean_rmse": np.mean(values) if metric == "rmse" else None,
                    "mean_se": np.mean(values) if metric == "se" else None,
                }
            )

        # Sort by mean value (lower is better for both SE and RMSE)
        estimator_means.sort(key=lambda x: x["mean_value"])
        rankings[scenario] = estimator_means

    return rankings


def create_method_comparison_matrix(
    rankings: Dict[str, List[Dict]], output_path: Path = None
) -> plt.Figure:
    """Create a heatmap showing method rankings across all scenarios."""

    # Get all unique methods
    all_methods = set()
    for ranking in rankings.values():
        for r in ranking:
            all_methods.add(r["estimator"])

    # Calculate average rank for each method
    method_avg_ranks = {}
    for method in all_methods:
        ranks = []
        for ranking in rankings.values():
            # Find this method's rank in this scenario
            for idx, r in enumerate(ranking):
                if r["estimator"] == method:
                    ranks.append(idx + 1)
                    break
            else:
                # Method not in this ranking (failed or not tested)
                ranks.append(len(ranking) + 1)  # Penalty rank
        method_avg_ranks[method] = np.mean(ranks)

    # Sort methods by average rank (best first)
    method_order = sorted(all_methods, key=lambda m: method_avg_ranks[m])

    # Sort scenarios
    scenario_order = []
    for n in [500, 1000, 2500, 5000]:
        for oracle in [5, 10, 25, 50, 100]:
            key = f"n={n}, oracle={oracle}%"
            if key in rankings:
                scenario_order.append(key)

    # Build ranking matrix
    rank_matrix = np.full((len(method_order), len(scenario_order)), np.nan)

    for j, scenario in enumerate(scenario_order):
        if scenario in rankings:
            ranking = rankings[scenario]
            # Create rank dict
            rank_dict = {r["estimator"]: idx + 1 for idx, r in enumerate(ranking)}

            for i, method in enumerate(method_order):
                if method in rank_dict:
                    rank_matrix[i, j] = rank_dict[method]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))

    # Use diverging colormap - blue (good) to red (bad)
    import seaborn as sns

    # Add average rank to y-axis labels
    y_labels = [
        f"{method} (avg: {method_avg_ranks[method]:.1f})" for method in method_order
    ]

    sns.heatmap(
        rank_matrix,
        annot=True,
        fmt=".0f",
        cmap="RdYlBu_r",
        xticklabels=[
            s.replace("n=", "").replace(", oracle=", "\n") for s in scenario_order
        ],
        yticklabels=y_labels,
        cbar_kws={"label": "Rank (1=best)"},
        vmin=1,
        vmax=15,
        ax=ax,
    )

    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Estimator", fontsize=12)
    ax.set_title(
        "Estimator Rankings Across All Scenarios", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {output_path}")

    return fig


def main() -> Dict[str, List[Dict]]:
    """Main analysis function."""

    print("=" * 70)
    print("ESTIMATOR COMPARISON ANALYSIS")
    print("=" * 70)

    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"Loaded {len(results)} successful experiments")

    # Analyze by scenario
    print("\nAnalyzing by scenario...")
    scenarios = analyze_by_scenario(results)
    print(f"Found {len(scenarios)} unique scenarios")

    # Compute rankings using standard error
    print("\nComputing rankings by standard error...")
    rankings = compute_rankings(scenarios, metric="se")

    # Print top performers for a few key scenarios
    key_scenarios = ["n=1000, oracle=50%", "n=5000, oracle=10%", "n=5000, oracle=100%"]

    for scenario in key_scenarios:
        if scenario in rankings:
            print(f"\n{scenario}:")
            print("  Top 5 estimators (by SE):")
            for i, r in enumerate(rankings[scenario][:5], 1):
                print(f"    {i}. {r['estimator']}: SE = {r['mean_value']:.5f}")

    # Create visualizations
    print("\nCreating visualization...")
    fig = create_method_comparison_matrix(
        rankings, Path("results/analysis/estimator_rankings_heatmap.png")
    )

    print("\n" + "=" * 70)
    print("ESTIMATOR COMPARISON COMPLETE")
    print("=" * 70)

    return rankings


if __name__ == "__main__":
    rankings = main()
