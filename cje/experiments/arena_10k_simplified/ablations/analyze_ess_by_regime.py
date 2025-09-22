#!/usr/bin/env python3
"""Analyze ESS by regime to show detailed breakdown."""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(file_path: str) -> list:
    """Load JSONL results file."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def main() -> None:
    # Load results
    results = load_results("results/all_experiments.jsonl")

    # Extract ESS data
    data = []
    for result in results:
        if not result.get("success", False):
            continue

        estimator = result["spec"]["estimator"]
        if estimator not in ["raw-ips", "calibrated-ips"]:
            continue

        sample_size = result["spec"]["sample_size"]
        oracle_coverage = result["spec"]["oracle_coverage"]
        ess_rel = result.get("ess_relative", {})

        for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
            if policy in ess_rel:
                data.append(
                    {
                        "estimator": estimator,
                        "sample_size": sample_size,
                        "oracle_coverage": oracle_coverage,
                        "policy": policy,
                        "ess_percent": ess_rel[policy],
                    }
                )

    df = pd.DataFrame(data)

    # Create pivot table by sample size
    print("\n=== ESS % by Sample Size (averaged across oracle coverage) ===\n")
    pivot_size = df.pivot_table(
        values="ess_percent",
        index=["policy", "sample_size"],
        columns="estimator",
        aggfunc="mean",
    ).round(1)
    print(pivot_size)

    # Create pivot table by oracle coverage
    print("\n=== ESS % by Oracle Coverage (averaged across sample sizes) ===\n")
    pivot_coverage = df.pivot_table(
        values="ess_percent",
        index=["policy", "oracle_coverage"],
        columns="estimator",
        aggfunc="mean",
    ).round(1)
    print(pivot_coverage)

    # Show specific regimes
    print("\n=== ESS % for Key Regimes ===\n")

    key_regimes = [
        (250, 0.05, "Small sample, low oracle"),
        (1000, 0.10, "Medium sample, medium oracle"),
        (5000, 0.50, "Large sample, high oracle"),
    ]

    for n, cov, desc in key_regimes:
        print(f"\n{desc} (n={n}, coverage={cov}):")
        regime_data = df[(df["sample_size"] == n) & (df["oracle_coverage"] == cov)]

        if len(regime_data) > 0:
            pivot = regime_data.pivot_table(
                values="ess_percent",
                index="policy",
                columns="estimator",
                aggfunc="mean",
            ).round(1)
            print(pivot)
        else:
            print("No data for this regime")

    # Show overall statistics
    print("\n=== Overall Statistics ===\n")
    overall = (
        df.groupby(["estimator", "policy"])["ess_percent"]
        .agg(["mean", "std", "min", "max"])
        .round(1)
    )
    print(overall)


if __name__ == "__main__":
    main()
