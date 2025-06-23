#!/usr/bin/env python3
"""Analyze human labels by policy using the correct mapping from internal tracking."""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict


def load_ratings() -> pd.DataFrame:
    """Load ratings from mturk_labels.csv"""
    ratings_df = pd.read_csv("data/labeling/mturk_labels.csv")
    print(f"Loaded {len(ratings_df)} ratings")
    return ratings_df


def load_actual_task_mapping() -> dict[str, str]:
    """Load the actual task to policy mapping from internal tracking"""
    task_to_policy = {}

    with open("data/labeling/internal_tracking.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            task_id = data["task_id"]
            policy = data["policy"]
            split = data.get("split", "")

            # Normalize policy names
            if policy == "pi_0":
                if split == "calibration":
                    policy = "π₀ (calibration)"
                else:
                    policy = "π₀ (evaluation)"
            elif policy == "pi_cot":
                policy = "π_cot"
            elif policy == "pi_bigger_model":
                policy = "π_bigger_model"
            elif policy == "pi_bad":
                policy = "π_bad"

            task_to_policy[task_id] = policy

    print(f"Loaded mapping for {len(task_to_policy)} tasks")
    return task_to_policy


def analyze_ratings_by_policy(
    ratings_df: pd.DataFrame, task_to_policy: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Calculate summary statistics by policy"""
    # Add policy column to ratings
    ratings_df["policy"] = ratings_df["task_id"].map(task_to_policy)

    # Remove any unmapped tasks
    ratings_df = ratings_df[ratings_df["policy"].notna()]

    # Group by policy and calculate statistics
    policy_stats = {}

    for policy in ratings_df["policy"].unique():
        policy_data = ratings_df[ratings_df["policy"] == policy]["rating"]

        policy_stats[policy] = {
            "count": len(policy_data),
            "mean": policy_data.mean(),
            "std": policy_data.std(),
            "median": policy_data.median(),
            "min": policy_data.min(),
            "max": policy_data.max(),
            "q25": policy_data.quantile(0.25),
            "q75": policy_data.quantile(0.75),
            "ratings_distribution": policy_data.value_counts().sort_index().to_dict(),
        }

    return ratings_df, policy_stats


def print_statistics(policy_stats: dict[str, dict[str, Any]]) -> None:
    """Print formatted statistics"""
    print("\n" + "=" * 80)
    print("HUMAN LABEL STATISTICS BY POLICY (CORRECTED)")
    print("=" * 80)

    # Expected counts based on experiment design
    # Note: internal tracking shows actual distribution

    for policy, stats in sorted(policy_stats.items()):
        print(f"\n{policy}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Median: {stats['median']:.1f}")
        print(f"  Range: [{stats['min']}, {stats['max']}]")
        print(f"  IQR: [{stats['q25']:.1f}, {stats['q75']:.1f}]")

        # Print rating distribution
        print("  Distribution:")
        for rating in range(1, 11):
            count = stats["ratings_distribution"].get(rating, 0)
            pct = count / stats["count"] * 100 if stats["count"] > 0 else 0
            bar = "█" * int(pct / 2)  # Scale to fit
            print(f"    {rating:2d}: {bar:<25} {count:4d} ({pct:5.1f}%)")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    total_count = sum(s["count"] for s in policy_stats.values())
    print(f"Total labels: {total_count}")


def create_visualization(ratings_df: pd.DataFrame) -> None:
    """Create visualization of ratings by policy"""
    # Filter to main policies for cleaner visualization
    main_policies = ["π₀ (calibration)", "π_cot", "π_bigger_model", "π_bad"]
    plot_df = ratings_df[ratings_df["policy"].isin(main_policies)].copy()

    # Set up the plot
    plt.figure(figsize=(15, 10))

    # Define color palette
    colors = {
        "π₀ (calibration)": "blue",
        "π_cot": "green",
        "π_bigger_model": "orange",
        "π_bad": "red",
    }

    # 1. Box plot of ratings by policy
    plt.subplot(2, 2, 1)
    box_plot = plot_df.boxplot(
        column="rating", by="policy", ax=plt.gca(), patch_artist=True
    )
    plt.title("Rating Distribution by Policy")
    plt.xlabel("Policy")
    plt.ylabel("Rating (1-10)")
    plt.xticks(rotation=45)
    plt.suptitle("")  # Remove automatic title

    # 2. Violin plot
    plt.subplot(2, 2, 2)
    sns.violinplot(data=plot_df, x="policy", y="rating", palette=colors)
    plt.title("Rating Density by Policy")
    plt.xlabel("Policy")
    plt.ylabel("Rating (1-10)")
    plt.xticks(rotation=45)

    # 3. Mean ratings with error bars
    plt.subplot(2, 2, 3)
    policy_means = plot_df.groupby("policy")["rating"].agg(["mean", "std", "count"])
    policy_means["sem"] = policy_means["std"] / np.sqrt(policy_means["count"])

    x_pos = range(len(policy_means))
    plt.bar(
        x_pos,
        policy_means["mean"],
        yerr=policy_means["sem"],
        capsize=5,
        color=[colors[p] for p in policy_means.index],
    )
    plt.xticks(x_pos, policy_means.index, rotation=45)
    plt.ylabel("Mean Rating")
    plt.title("Mean Ratings by Policy (with SEM)")
    plt.ylim(0, 10)

    # Add significance indicators
    plt.axhline(
        y=policy_means.loc["π₀ (calibration)", "mean"],
        color="blue",
        linestyle="--",
        alpha=0.5,
        label="π₀ baseline",
    )

    # 4. Sample size by policy
    plt.subplot(2, 2, 4)
    policy_counts = plot_df["policy"].value_counts()
    policy_counts.plot(kind="bar", color=[colors[p] for p in policy_counts.index])
    plt.title("Number of Labels by Policy")
    plt.xlabel("Policy")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(
        "scripts/policy_ratings_analysis_corrected.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("\nVisualization saved to: scripts/policy_ratings_analysis_corrected.png")


def analyze_policy_differences(ratings_df: pd.DataFrame) -> None:
    """Analyze statistical differences between policies"""
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS")
    print("=" * 80)

    from scipy import stats

    # Focus on main policies
    main_policies = ["π₀ (calibration)", "π_cot", "π_bigger_model", "π_bad"]
    policy_ratings = {}

    for policy in main_policies:
        if policy in ratings_df["policy"].values:
            policy_ratings[policy] = ratings_df[ratings_df["policy"] == policy][
                "rating"
            ].values

    # Perform pairwise comparisons
    print("\nPairwise Mann-Whitney U tests (p-values):")
    print("(Lower p-values indicate more significant differences)")

    comparisons = []
    for i, policy1 in enumerate(main_policies):
        if policy1 not in policy_ratings:
            continue
        for policy2 in main_policies[i + 1 :]:
            if policy2 not in policy_ratings:
                continue

            stat, p_value = stats.mannwhitneyu(
                policy_ratings[policy1],
                policy_ratings[policy2],
                alternative="two-sided",
            )

            # Calculate effect size (rank-biserial correlation)
            n1 = len(policy_ratings[policy1])
            n2 = len(policy_ratings[policy2])
            r = 1 - (2 * stat) / (n1 * n2)

            # Mean difference
            mean_diff = policy_ratings[policy1].mean() - policy_ratings[policy2].mean()

            comparisons.append(
                {
                    "comparison": f"{policy1} vs {policy2}",
                    "p_value": p_value,
                    "effect_size": r,
                    "mean_diff": mean_diff,
                    "n1": n1,
                    "n2": n2,
                }
            )

            print(f"\n  {policy1} vs {policy2}:")
            print(
                f"    p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}"
            )
            print(f"    Effect size (r): {r:.3f}")
            print(f"    Mean difference: {mean_diff:.2f}")
            print(f"    Sample sizes: {n1} vs {n2}")

    # Kruskal-Wallis test (overall)
    print("\n\nKruskal-Wallis test (overall difference):")
    all_groups = [policy_ratings[p] for p in main_policies if p in policy_ratings]
    h_stat, p_value = stats.kruskal(*all_groups)
    print(f"  H-statistic: {h_stat:.2f}")
    print(
        f"  p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}"
    )

    # Don't return comparisons since the function is void


def main() -> None:
    print("Analyzing human labels by policy (with correct mapping)...")

    # Load ratings
    ratings_df = load_ratings()

    # Load actual task to policy mapping
    task_to_policy = load_actual_task_mapping()

    # Analyze ratings by policy
    ratings_df, policy_stats = analyze_ratings_by_policy(ratings_df, task_to_policy)

    # Print statistics
    print_statistics(policy_stats)

    # Create visualization
    create_visualization(ratings_df)

    # Analyze differences
    analyze_policy_differences(ratings_df)

    # Save detailed results
    results_json: Dict[str, Any] = {
        "summary_statistics": {},
        "total_labels": int(len(ratings_df)),
        "unique_tasks": int(ratings_df["task_id"].nunique()),
        "statistical_comparisons": [],  # We don't have comparisons anymore
    }

    for policy, stats in policy_stats.items():
        results_json["summary_statistics"][policy] = {
            "count": int(stats["count"]),
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "median": float(stats["median"]),
            "min": int(stats["min"]),
            "max": int(stats["max"]),
            "q25": float(stats["q25"]),
            "q75": float(stats["q75"]),
            "ratings_distribution": {
                str(k): int(v) for k, v in stats["ratings_distribution"].items()
            },
        }

    with open("scripts/policy_analysis_results_corrected.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print("\nDetailed results saved to: scripts/policy_analysis_results_corrected.json")


if __name__ == "__main__":
    main()
