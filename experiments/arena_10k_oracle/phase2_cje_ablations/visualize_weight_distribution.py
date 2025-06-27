#!/usr/bin/env python3
"""Visualize the importance weight distributions to understand pi_bad's victory."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_data():
    """Load the teacher forcing data."""
    tf_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    data = []
    with open(tf_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def compute_weights_by_length(data):
    """Compute importance weights grouped by response length."""
    # Categories
    length_bins = {
        "Short (<100)": (0, 100),
        "Medium (100-500)": (100, 500),
        "Long (>500)": (500, float("inf")),
    }

    # Store weights by policy and length category
    weights_by_category = defaultdict(lambda: defaultdict(list))

    for item in data:
        response_len = len(item["response"])
        p0_logp = item["total_logprob"]

        # Determine category
        category = None
        for cat_name, (min_len, max_len) in length_bins.items():
            if min_len <= response_len < max_len:
                category = cat_name
                break

        if category:
            for policy, target_logp in item.get("target_logps", {}).items():
                weight = np.exp(np.clip(target_logp - p0_logp, -20, 20))
                weights_by_category[policy][category].append(weight)

    return weights_by_category, length_bins


def plot_weight_distributions(weights_by_category):
    """Create visualization of weight distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Importance Weight Distributions by Response Length", fontsize=16)

    policies = ["pi_bad", "pi_bigger_model", "pi_cot"]
    categories = ["Short (<100)", "Medium (100-500)", "Long (>500)"]
    colors = {"pi_bad": "red", "pi_bigger_model": "blue", "pi_cot": "green"}

    for idx, category in enumerate(categories):
        ax = axes[idx]
        ax.set_title(category)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Average Weight (log scale)")
        ax.set_yscale("log")

        avg_weights = []
        labels = []
        bar_colors = []

        for policy in policies:
            weights = weights_by_category[policy][category]
            if weights:
                avg_weight = np.mean(weights)
                avg_weights.append(avg_weight)
                labels.append(policy)
                bar_colors.append(colors[policy])

        bars = ax.bar(range(len(labels)), avg_weights, color=bar_colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)

        # Add value labels on bars
        for bar, weight in zip(bars, avg_weights):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{weight:.1e}" if weight > 1000 else f"{weight:.1f}",
                ha="center",
                va="bottom",
            )

        # Add sample count
        for i, policy in enumerate(policies):
            count = len(weights_by_category[policy][category])
            ax.text(
                i,
                0.5,
                f"n={count}",
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
            )

    plt.tight_layout()
    plt.savefig("weight_distributions.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to weight_distributions.png")


def analyze_p0_failures(data):
    """Analyze samples with P0 logp = -50.0."""
    suspect_samples = [item for item in data if item["total_logprob"] == -50.0]

    print(f"\nAnalysis of P0 Scoring Failures:")
    print(f"Total samples with P0 logp = -50.0: {len(suspect_samples)}")

    # Group by response length
    length_dist = defaultdict(int)
    for item in suspect_samples:
        response_len = len(item["response"])
        if response_len < 100:
            length_dist["Short"] += 1
        elif response_len < 500:
            length_dist["Medium"] += 1
        else:
            length_dist["Long"] += 1

    print("\nDistribution by response length:")
    for category, count in sorted(length_dist.items()):
        print(f"  {category}: {count} ({count/len(suspect_samples)*100:.1f}%)")

    # Check impact on extreme weights
    extreme_weights = 0
    for item in suspect_samples:
        p0_logp = item["total_logprob"]
        for policy, target_logp in item.get("target_logps", {}).items():
            weight = np.exp(np.clip(target_logp - p0_logp, -20, 20))
            if weight > 1e8:
                extreme_weights += 1

    print(f"\nExtreme weights (>1e8) from P0 failures: {extreme_weights}")


def main():
    """Run the visualization."""
    print("Loading data...")
    data = load_data()

    print(f"Loaded {len(data)} samples")

    # Compute weights by length
    weights_by_category, length_bins = compute_weights_by_length(data)

    # Create visualization
    print("\nCreating weight distribution visualization...")
    plot_weight_distributions(weights_by_category)

    # Analyze P0 failures
    analyze_p0_failures(data)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for policy in ["pi_bad", "pi_bigger_model", "pi_cot"]:
        print(f"\n{policy}:")
        for category in ["Short (<100)", "Medium (100-500)", "Long (>500)"]:
            weights = weights_by_category[policy][category]
            if weights:
                print(
                    f"  {category}: n={len(weights)}, avg={np.mean(weights):.2e}, "
                    f"median={np.median(weights):.2e}, max={np.max(weights):.2e}"
                )


if __name__ == "__main__":
    main()
