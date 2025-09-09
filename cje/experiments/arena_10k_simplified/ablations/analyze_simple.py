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
    results = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("success"):
                results.append(data)
    return results


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
        # Handle both old and new parameter names for backward compatibility
        use_cal = extra.get(
            "use_weight_calibration",
            extra.get("use_calibration", spec.get("use_calibration", False)),
        )
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

        # Add policy-specific errors and robust SEs
        if "estimates" in r and "oracle_truths" in r:
            for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
                if policy in r["estimates"] and policy in r["oracle_truths"]:
                    row[f"error_{policy}"] = abs(
                        r["estimates"][policy] - r["oracle_truths"][policy]
                    )

        # Add robust standard errors (if available)
        if "robust_standard_errors" in r:
            for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
                if policy in r["robust_standard_errors"]:
                    row[f"robust_se_{policy}"] = r["robust_standard_errors"][policy]
        elif "standard_errors" in r:
            # Fallback to standard errors if robust not available
            for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
                if policy in r["standard_errors"]:
                    row[f"robust_se_{policy}"] = r["standard_errors"][policy]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Debug: Show what estimators are in the data
    print("\nEstimators found in data:")
    print(df["estimator"].value_counts())

    # 1. Main comparison table
    print("\n" + "=" * 80)
    print("MAIN RESULTS BY ESTIMATOR")
    print("=" * 80)

    summary = (
        df.groupby("estimator")
        .agg({"rmse": ["mean", "std", "min", "max", "count"], "runtime": "mean"})
        .round(4)
    )
    print(summary)

    # 2. Calibration impact
    print("\n" + "=" * 80)
    print("CALIBRATION IMPACT")
    print("=" * 80)

    # Get all estimators that have both calibration on/off in the data
    estimators_with_cal_choice = []
    for estimator in df["estimator"].unique():
        est_df = df[df["estimator"] == estimator]
        has_cal_on = any(est_df["use_cal"] == True)
        has_cal_off = any(est_df["use_cal"] == False)
        if has_cal_on and has_cal_off:
            estimators_with_cal_choice.append(estimator)

    for method in sorted(estimators_with_cal_choice):
        method_df = df[df["estimator"] == method]
        cal_on_df = method_df[method_df["use_cal"] == True]["rmse"].dropna()
        cal_off_df = method_df[method_df["use_cal"] == False]["rmse"].dropna()

        if len(cal_on_df) > 0 and len(cal_off_df) > 0:
            cal_on = cal_on_df.mean()
            cal_off = cal_off_df.mean()
            improvement = (cal_off - cal_on) / cal_off * 100 if cal_off > 0 else 0
            print(
                f"{method:20s}: Without cal={cal_off:.4f}, With cal={cal_on:.4f}, Improvement={improvement:.1f}%"
            )

    # 3. IIC impact
    print("\n" + "=" * 80)
    print("IIC IMPACT (all methods)")
    print("=" * 80)

    # Check all estimators - IIC should have identical results if properly implemented
    all_estimators = sorted(df["estimator"].unique())

    for method in all_estimators:
        method_df = df[df["estimator"] == method]
        iic_on_df = method_df[method_df["use_iic"] == True]["rmse"].dropna()
        iic_off_df = method_df[method_df["use_iic"] == False]["rmse"].dropna()

        if len(iic_on_df) > 0 and len(iic_off_df) > 0:
            iic_on = iic_on_df.mean()
            iic_off = iic_off_df.mean()
            # IIC should NOT affect RMSE (only SEs), so difference should be ~0
            diff = iic_on - iic_off
            print(
                f"{method:20s}: Without IIC={iic_off:.4f}, With IIC={iic_on:.4f}, Diff={diff:+.4f}"
            )
        elif len(iic_on_df) > 0:
            print(f"{method:20s}: Only IIC=True data, RMSE={iic_on_df.mean():.4f}")
        elif len(iic_off_df) > 0:
            print(f"{method:20s}: Only IIC=False data, RMSE={iic_off_df.mean():.4f}")

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

    # 9. Robust Standard Errors Summary
    print("\n" + "=" * 80)
    print("ROBUST STANDARD ERRORS (mean across experiments)")
    print("=" * 80)

    se_cols = [c for c in df.columns if c.startswith("robust_se_")]
    if se_cols:
        se_summary = df.groupby("estimator")[se_cols].mean()
        se_summary.columns = [c.replace("robust_se_", "") for c in se_summary.columns]
        print(se_summary.round(4))

        # Show SE comparison for IIC on/off with actual IIC diagnostics
        print("\n" + "-" * 40)
        print("IIC Impact on Robust SEs (mean SEs)")
        print("-" * 40)

        # First show SE impact
        for estimator in sorted(df["estimator"].unique()):
            est_df = df[df["estimator"] == estimator]
            iic_on = est_df[est_df["use_iic"] == True][se_cols].mean().mean()
            iic_off = est_df[est_df["use_iic"] == False][se_cols].mean().mean()
            if not pd.isna(iic_on) and not pd.isna(iic_off):
                reduction = (iic_off - iic_on) / iic_off * 100 if iic_off > 0 else 0
                print(
                    f"{estimator:20s}: Without IIC={iic_off:.4f}, With IIC={iic_on:.4f}, Reduction={reduction:.1f}%"
                )

        # Extract and show IIC effectiveness from diagnostics
        print("\n" + "-" * 40)
        print("IIC Effectiveness (from diagnostics)")
        print("-" * 40)
        iic_effectiveness: Dict[str, Any] = {}
        for r in results:
            if r.get("success") and "iic_diagnostics" in r:
                est = r["spec"]["estimator"]
                if est not in iic_effectiveness:
                    iic_effectiveness[est] = {"r_squared": [], "var_reduction": []}

                # Average across policies
                for policy, diag in r["iic_diagnostics"].items():
                    if isinstance(diag, dict):
                        iic_effectiveness[est]["r_squared"].append(
                            diag.get("r_squared", 0)
                        )
                        iic_effectiveness[est]["var_reduction"].append(
                            diag.get("var_reduction", 0)
                        )

        for est in sorted(iic_effectiveness.keys()):
            r2_mean = np.mean(iic_effectiveness[est]["r_squared"])
            var_red_mean = np.mean(iic_effectiveness[est]["var_reduction"])
            print(f"{est:20s}: R²={r2_mean:.3f}, Var reduction={var_red_mean:.1%}")

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
