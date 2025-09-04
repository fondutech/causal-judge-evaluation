#!/usr/bin/env python3
"""
Simple analysis of ablation results that actually works.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_results(path: str = "results/all_experiments.jsonl") -> List[Dict]:
    """Load experiment results."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if json.loads(line).get("success")]


def main() -> None:
    """Run simple analysis."""
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} successful experiments")

    # Convert to DataFrame for easier analysis
    rows = []
    for r in results:
        spec = r["spec"]
        # Extract parameters from extra or directly from spec
        extra = spec.get("extra", {})
        use_cal = extra.get("use_calibration", spec.get("use_calibration", False))
        use_iic = extra.get("use_iic", spec.get("use_iic", False))
        weight_mode = extra.get("weight_mode", "hajek")
        reward_calib_mode = extra.get("reward_calibration_mode", "auto")

        row = {
            "estimator": spec["estimator"],
            "n": spec["sample_size"],
            "oracle_pct": spec["oracle_coverage"],
            "use_cal": use_cal,
            "use_iic": use_iic,
            "weight_mode": weight_mode,
            "reward_calib_mode": reward_calib_mode,
            "seed": r.get("seed", spec.get("seed_base", "N/A")),
            "rmse": r.get("rmse_vs_oracle", np.nan),
            "runtime": r.get("runtime_s", 0),
        }

        # Add policy-specific errors (including unhelpful for individual analysis)
        if "estimates" in r and "oracle_truths" in r:
            for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
                if policy in r["estimates"] and policy in r["oracle_truths"]:
                    row[f"error_{policy}"] = abs(
                        r["estimates"][policy] - r["oracle_truths"][policy]
                    )

        rows.append(row)

    df = pd.DataFrame(rows)

    # 1. Main comparison table
    print("\n" + "=" * 80)
    print("MAIN RESULTS BY ESTIMATOR")
    print("=" * 80)

    summary = (
        df.groupby("estimator")
        .agg({"rmse": ["mean", "std", "min", "max"], "runtime": "mean"})
        .round(4)
    )
    print(summary)

    # 2. Calibration impact
    print("\n" + "=" * 80)
    print("CALIBRATION IMPACT")
    print("=" * 80)

    # Compare all methods with/without calibration
    all_methods = ["ips", "dr-cpo", "stacked-dr"]  # removed tmle, mrdr from config
    for method in all_methods:
        method_df = df[df["estimator"] == method]
        if not method_df.empty:
            cal_on = method_df[method_df["use_cal"] == True]["rmse"].mean()
            cal_off = method_df[method_df["use_cal"] == False]["rmse"].mean()
            improvement = (cal_off - cal_on) / cal_off * 100
            print(
                f"{method:15s}: Without cal={cal_off:.4f}, With cal={cal_on:.4f}, Improvement={improvement:.1f}%"
            )

    # 3. IIC impact
    print("\n" + "=" * 80)
    print("IIC IMPACT (all methods)")
    print("=" * 80)

    for method in all_methods:
        method_df = df[df["estimator"] == method]
        if not method_df.empty:
            iic_on = method_df[method_df["use_iic"] == True]["rmse"].mean()
            iic_off = method_df[method_df["use_iic"] == False]["rmse"].mean()
            improvement = (iic_off - iic_on) / iic_off * 100 if iic_off > 0 else 0
            print(
                f"{method:15s}: Without IIC={iic_off:.4f}, With IIC={iic_on:.4f}, Improvement={improvement:.1f}%"
            )

    # 4. Sample size scaling
    print("\n" + "=" * 80)
    print("SAMPLE SIZE SCALING")
    print("=" * 80)

    size_summary = df.groupby(["estimator", "n"])["rmse"].mean().unstack()
    print(size_summary.round(4))

    # 5. Oracle coverage impact
    print("\n" + "=" * 80)
    print("ORACLE COVERAGE IMPACT")
    print("=" * 80)

    oracle_summary = df.groupby(["estimator", "oracle_pct"])["rmse"].mean().unstack()
    print(oracle_summary.round(4))

    # 6. Weight mode impact (Hajek vs Raw)
    print("\n" + "=" * 80)
    print("WEIGHT MODE IMPACT (Hajek vs Raw)")
    print("=" * 80)

    all_methods = df["estimator"].unique()
    for method in all_methods:
        method_df = df[df["estimator"] == method]
        if not method_df.empty:
            hajek = method_df[method_df["weight_mode"] == "hajek"]["rmse"].mean()
            raw = method_df[method_df["weight_mode"] == "raw"]["rmse"].mean()
            if not pd.isna(hajek) and not pd.isna(raw):
                diff = raw - hajek
                print(
                    f"{method:15s}: Hajek={hajek:.4f}, Raw={raw:.4f}, Diff={diff:+.4f}"
                )

    # 7. Reward calibration mode impact
    print("\n" + "=" * 80)
    print("REWARD CALIBRATION MODE IMPACT")
    print("=" * 80)

    calib_summary = (
        df.groupby(["estimator", "reward_calib_mode"])["rmse"].mean().unstack()
    )
    if not calib_summary.empty:
        print(calib_summary.round(4))

    # 8. Policy-specific performance
    print("\n" + "=" * 80)
    print("POLICY-SPECIFIC ERRORS (mean absolute error)")
    print(
        "Note: 'unhelpful' is shown but excluded from aggregated RMSE due to different distribution"
    )
    print("=" * 80)

    policy_cols = [c for c in df.columns if c.startswith("error_")]
    if policy_cols:
        policy_summary = df.groupby("estimator")[policy_cols].mean()
        policy_summary.columns = [
            c.replace("error_", "") for c in policy_summary.columns
        ]
        print(policy_summary.round(4))

    # 7. Best configurations
    print("\n" + "=" * 80)
    print("TOP 10 BEST CONFIGURATIONS")
    print("=" * 80)

    best = df.nsmallest(10, "rmse")[
        ["estimator", "n", "oracle_pct", "use_cal", "use_iic", "rmse"]
    ]
    print(best.to_string(index=False))

    # 8. Save summary tables
    output_dir = Path("results/analysis")
    output_dir.mkdir(exist_ok=True)

    # Save main summary
    summary.to_csv(output_dir / "main_summary.csv")
    print(f"\nSaved summary to {output_dir}/main_summary.csv")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
