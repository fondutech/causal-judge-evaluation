#!/usr/bin/env python3
"""
Detailed analysis of calibration impact with proper statistical testing.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any, Tuple


def load_results(path: str = "results/all_experiments.jsonl") -> List[Dict]:
    """Load experiment results."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if json.loads(line).get("success")]


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)

    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_val * se

    return (mean - margin, mean + margin)


def paired_t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Perform paired t-test if data can be paired, otherwise Welch's t-test."""
    if len(group1) == len(group2):
        # Try paired test
        t_stat, p_value = stats.ttest_rel(group1, group2)
    else:
        # Fall back to Welch's t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    return t_stat, p_value


def main() -> None:
    """Run detailed calibration analysis."""
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} successful experiments\n")

    # Convert to DataFrame
    rows = []
    for r in results:
        spec = r["spec"]
        extra = spec.get("extra", {})
        # Handle both old and new parameter names for backward compatibility
        use_cal = extra.get(
            "use_weight_calibration",
            extra.get("use_calibration", spec.get("use_calibration", False)),
        )
        use_iic = extra.get("use_iic", spec.get("use_iic", False))

        row = {
            "estimator": spec["estimator"],
            "n": spec["sample_size"],
            "oracle_pct": spec["oracle_coverage"],
            "use_cal": use_cal,
            "use_iic": use_iic,
            "seed": r["seed"],
            "rmse": r.get("rmse_vs_oracle", np.nan),
        }

        # Add individual estimates and standard errors if available
        if "estimates" in r:
            for policy, value in r["estimates"].items():
                row[f"est_{policy}"] = value

        if "standard_errors" in r:
            for policy, value in r["standard_errors"].items():
                row[f"se_{policy}"] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    # Focus on DR methods
    dr_methods = ["dr-cpo", "tmle", "mrdr", "stacked-dr"]

    print("=" * 80)
    print("CALIBRATION IMPACT ON DR METHODS - DETAILED ANALYSIS")
    print("=" * 80)

    for method in dr_methods:
        method_df = df[df["estimator"] == method]
        if method_df.empty:
            continue

        print(f"\n{method.upper()}")
        print("-" * 40)

        # Get calibration on/off data
        cal_on = method_df[method_df["use_cal"] == True]
        cal_off = method_df[method_df["use_cal"] == False]

        if cal_on.empty or cal_off.empty:
            print("  Missing calibration on/off data")
            continue

        # RMSE analysis
        rmse_on = cal_on["rmse"].dropna().values
        rmse_off = cal_off["rmse"].dropna().values

        mean_on = np.mean(rmse_on)
        mean_off = np.mean(rmse_off)
        std_on = np.std(rmse_on, ddof=1)
        std_off = np.std(rmse_off, ddof=1)

        ci_on = compute_confidence_interval(rmse_on)
        ci_off = compute_confidence_interval(rmse_off)

        print(f"  Without calibration:")
        print(f"    Mean RMSE: {mean_off:.5f} ± {std_off:.5f}")
        print(f"    95% CI: [{ci_off[0]:.5f}, {ci_off[1]:.5f}]")
        print(f"    N samples: {len(rmse_off)}")

        print(f"  With calibration:")
        print(f"    Mean RMSE: {mean_on:.5f} ± {std_on:.5f}")
        print(f"    95% CI: [{ci_on[0]:.5f}, {ci_on[1]:.5f}]")
        print(f"    N samples: {len(rmse_on)}")

        # Improvement analysis
        improvement_pct = (mean_off - mean_on) / mean_off * 100 if mean_off > 0 else 0
        print(f"  Improvement: {improvement_pct:.2f}%")

        # Statistical test
        if len(rmse_on) > 1 and len(rmse_off) > 1:
            t_stat, p_value = paired_t_test(rmse_off, rmse_on)
            print(f"  Statistical test: t={t_stat:.3f}, p={p_value:.4f}")

            if p_value < 0.05:
                if mean_on < mean_off:
                    print(
                        f"  ✓ Calibration SIGNIFICANTLY IMPROVES performance (p<0.05)"
                    )
                else:
                    print(
                        f"  ✗ Calibration SIGNIFICANTLY DEGRADES performance (p<0.05)"
                    )
            else:
                print(f"  ~ No significant difference (p={p_value:.3f})")

        # Check if CIs overlap
        if ci_on[0] <= ci_off[1] and ci_off[0] <= ci_on[1]:
            print(f"  Note: Confidence intervals overlap")

    print("\n" + "=" * 80)
    print("IIC IMPACT ON DR METHODS")
    print("=" * 80)

    for method in dr_methods:
        method_df = df[df["estimator"] == method]
        if method_df.empty:
            continue

        print(f"\n{method.upper()}")
        print("-" * 40)

        # Get IIC on/off data (only for samples with calibration on)
        method_cal = method_df[method_df["use_cal"] == True]
        iic_on = method_cal[method_cal["use_iic"] == True]
        iic_off = method_cal[method_cal["use_iic"] == False]

        if iic_on.empty or iic_off.empty:
            print("  Missing IIC on/off data")
            continue

        rmse_iic_on = iic_on["rmse"].dropna().values
        rmse_iic_off = iic_off["rmse"].dropna().values

        mean_iic_on = np.mean(rmse_iic_on)
        mean_iic_off = np.mean(rmse_iic_off)

        improvement_pct = (
            (mean_iic_off - mean_iic_on) / mean_iic_off * 100 if mean_iic_off > 0 else 0
        )

        print(f"  Without IIC: {mean_iic_off:.5f}")
        print(f"  With IIC: {mean_iic_on:.5f}")
        print(f"  Improvement: {improvement_pct:.2f}%")

        if len(rmse_iic_on) > 1 and len(rmse_iic_off) > 1:
            t_stat, p_value = paired_t_test(rmse_iic_off, rmse_iic_on)
            print(f"  Statistical test: t={t_stat:.3f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(
                    f"  {'✓ Significant' if mean_iic_on < mean_iic_off else '✗ Significant degradation'}"
                )
            else:
                print(f"  ~ No significant difference")

    print("\n" + "=" * 80)
    print("BREAKDOWN BY SAMPLE SIZE AND ORACLE COVERAGE")
    print("=" * 80)

    # Check if calibration helps more at certain sample sizes
    for method in ["stacked-dr"]:  # Focus on best DR method
        method_df = df[df["estimator"] == method]
        if method_df.empty:
            continue

        print(f"\n{method.upper()} - Calibration impact by scenario")
        print("-" * 60)

        for n in sorted(method_df["n"].unique()):
            for oracle_pct in sorted(method_df["oracle_pct"].unique()):
                scenario_df = method_df[
                    (method_df["n"] == n) & (method_df["oracle_pct"] == oracle_pct)
                ]

                cal_on = scenario_df[scenario_df["use_cal"] == True]["rmse"].dropna()
                cal_off = scenario_df[scenario_df["use_cal"] == False]["rmse"].dropna()

                if len(cal_on) > 0 and len(cal_off) > 0:
                    mean_on = cal_on.mean()
                    mean_off = cal_off.mean()
                    improvement = (
                        (mean_off - mean_on) / mean_off * 100 if mean_off > 0 else 0
                    )

                    print(
                        f"  n={n:4d}, oracle={oracle_pct:4.0%}: "
                        f"Cal off={mean_off:.4f}, Cal on={mean_on:.4f}, "
                        f"Δ={improvement:+.1f}%"
                    )

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Compute overall statistics
    for method in dr_methods:
        method_df = df[df["estimator"] == method]
        if method_df.empty:
            continue

        cal_on = method_df[method_df["use_cal"] == True]["rmse"].dropna()
        cal_off = method_df[method_df["use_cal"] == False]["rmse"].dropna()

        if len(cal_on) > 0 and len(cal_off) > 0:
            mean_diff = cal_on.mean() - cal_off.mean()
            pct_change = (mean_diff / cal_off.mean()) * 100

            print(
                f"{method:12s}: Calibration changes RMSE by {mean_diff:+.5f} ({pct_change:+.1f}%)"
            )

    print("\nNote: Negative values mean calibration improves (reduces) RMSE")

    # Save detailed results
    output_dir = Path("results/analysis")
    output_dir.mkdir(exist_ok=True, parents=True)

    summary_df = (
        df.groupby(["estimator", "use_cal", "use_iic"])
        .agg({"rmse": ["mean", "std", "count"]})
        .round(5)
    )

    summary_df.to_csv(output_dir / "calibration_detail.csv")
    print(f"\nDetailed results saved to {output_dir}/calibration_detail.csv")


if __name__ == "__main__":
    main()
