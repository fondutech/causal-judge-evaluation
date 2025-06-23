#!/usr/bin/env python3
"""Analyze oracle label results by policy."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
from scipy import stats


def analyze_oracle_labels() -> None:
    """Analyze oracle labeling results."""

    # Load oracle labels
    oracle_df = pd.read_csv("data/labeling/oracle_labels.csv")

    print("=" * 80)
    print("ORACLE LABEL ANALYSIS")
    print("=" * 80)

    # Overall statistics
    print(f"\nTotal oracle labels: {len(oracle_df)}")
    print(f"Dataset types: {oracle_df['dataset_type'].value_counts().to_dict()}")

    # Analyze by policy
    print("\n" + "=" * 80)
    print("SCORES BY POLICY")
    print("=" * 80)

    policy_stats = []
    for policy in sorted(oracle_df["policy"].unique()):
        policy_data = oracle_df[oracle_df["policy"] == policy]
        scores = policy_data["oracle_score"].values

        policy_stat = {
            "policy": policy,
            "count": len(scores),
            "mean": np.mean(scores),
            "std": np.std(scores),
            "median": np.median(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "q25": np.percentile(scores, 25),
            "q75": np.percentile(scores, 75),
        }
        policy_stats.append(policy_stat)

        print(f"\n{policy}:")
        print(f"  Count: {policy_stat['count']}")
        print(f"  Mean: {policy_stat['mean']:.3f} ({policy_stat['mean']*10:.1f}/10)")
        print(f"  Std: {policy_stat['std']:.3f}")
        print(f"  Median: {policy_stat['median']:.3f}")
        print(f"  Range: [{policy_stat['min']:.2f}, {policy_stat['max']:.2f}]")
        print(f"  IQR: [{policy_stat['q25']:.2f}, {policy_stat['q75']:.2f}]")

    # Create comparison table
    stats_df = pd.DataFrame(policy_stats)

    # Separate by dataset type
    print("\n" + "=" * 80)
    print("CALIBRATION VS VALIDATION ANALYSIS")
    print("=" * 80)

    # Calibration analysis (pi_0 only)
    cal_data = oracle_df[oracle_df["dataset_type"] == "calibration"]
    if len(cal_data) > 0:
        print(f"\nCalibration Set (pi_0):")
        print(f"  Count: {len(cal_data)}")
        print(
            f"  Mean: {cal_data['oracle_score'].mean():.3f} ({cal_data['oracle_score'].mean()*10:.1f}/10)"
        )
        print(f"  Std: {cal_data['oracle_score'].std():.3f}")

    # Validation analysis (target policies)
    val_data = oracle_df[oracle_df["dataset_type"] == "validation"]
    if len(val_data) > 0:
        print(f"\nValidation Set:")
        for policy in sorted(val_data["policy"].unique()):
            policy_val = val_data[val_data["policy"] == policy]
            print(f"\n  {policy}:")
            print(f"    Count: {len(policy_val)}")
            print(
                f"    Mean: {policy_val['oracle_score'].mean():.3f} ({policy_val['oracle_score'].mean()*10:.1f}/10)"
            )
            print(f"    Std: {policy_val['oracle_score'].std():.3f}")

    # Statistical comparisons
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISONS")
    print("=" * 80)

    # Compare pi_bad to other policies
    if "pi_bad" in oracle_df["policy"].values:
        pi_bad_scores = oracle_df[oracle_df["policy"] == "pi_bad"]["oracle_score"]

        for policy in ["pi_0", "pi_cot", "pi_bigger_model"]:
            if policy in oracle_df["policy"].values:
                other_scores = oracle_df[oracle_df["policy"] == policy]["oracle_score"]

                # T-test
                t_stat, p_value = stats.ttest_ind(pi_bad_scores, other_scores)

                print(f"\npi_bad vs {policy}:")
                print(
                    f"  Mean difference: {pi_bad_scores.mean() - other_scores.mean():.3f}"
                )
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.3e}")
                print(f"  Significant: {'YES' if p_value < 0.001 else 'NO'}")

    # Visualize if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 8))

        # Box plot by policy
        plt.subplot(2, 2, 1)
        oracle_df.boxplot(column="oracle_score_10", by="policy", ax=plt.gca())
        plt.title("Oracle Scores by Policy")
        plt.ylabel("Score (1-10 scale)")
        plt.suptitle("")  # Remove automatic title

        # Histogram by policy
        plt.subplot(2, 2, 2)
        for policy in sorted(oracle_df["policy"].unique()):
            policy_data = oracle_df[oracle_df["policy"] == policy]["oracle_score_10"]
            plt.hist(policy_data, alpha=0.5, label=policy, bins=20)
        plt.xlabel("Score (1-10 scale)")
        plt.ylabel("Count")
        plt.title("Score Distribution by Policy")
        plt.legend()

        # Violin plot
        plt.subplot(2, 2, 3)
        sns.violinplot(data=oracle_df, x="policy", y="oracle_score_10")
        plt.title("Score Distribution (Violin Plot)")
        plt.ylabel("Score (1-10 scale)")

        # Mean comparison
        plt.subplot(2, 2, 4)
        mean_scores = oracle_df.groupby("policy")["oracle_score_10"].agg(
            ["mean", "sem"]
        )
        policies = mean_scores.index
        means = mean_scores["mean"]
        errors = mean_scores["sem"] * 1.96  # 95% CI

        plt.bar(policies, means, yerr=errors, capsize=5)
        plt.ylabel("Mean Score (1-10 scale)")
        plt.title("Mean Scores with 95% CI")
        plt.ylim(0, 10)

        # Add value labels
        for i, (policy, mean) in enumerate(zip(policies, means)):
            plt.text(i, mean + errors[i] + 0.1, f"{mean:.1f}", ha="center")

        plt.tight_layout()
        plt.savefig("data/labeling/oracle_analysis.png", dpi=300, bbox_inches="tight")
        print(f"\n✅ Visualization saved to: data/labeling/oracle_analysis.png")

    except ImportError:
        print("\n⚠️  Matplotlib not available, skipping visualization")

    # Save detailed results
    results = {
        "summary_stats": stats_df.to_dict("records"),
        "total_labels": len(oracle_df),
        "dataset_breakdown": oracle_df["dataset_type"].value_counts().to_dict(),
        "policy_means": {
            p: float(oracle_df[oracle_df["policy"] == p]["oracle_score"].mean())
            for p in oracle_df["policy"].unique()
        },
    }

    with open("data/labeling/oracle_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Detailed results saved to: data/labeling/oracle_analysis_results.json")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if "pi_bad" in results["policy_means"]:
        pi_bad_mean = results["policy_means"]["pi_bad"]
        other_means = [v for k, v in results["policy_means"].items() if k != "pi_bad"]

        if all(pi_bad_mean < other for other in other_means):
            print("✅ SUCCESS: pi_bad scored lower than all other policies!")
            print(f"   pi_bad mean: {pi_bad_mean:.3f} ({pi_bad_mean*10:.1f}/10)")
            print(f"   Other policy means: {[f'{m:.3f}' for m in sorted(other_means)]}")
        else:
            print("❌ ISSUE: pi_bad did not score lowest")
            print(f"   pi_bad mean: {pi_bad_mean:.3f}")
            print(f"   Other means: {sorted(other_means)}")


if __name__ == "__main__":
    analyze_oracle_labels()
