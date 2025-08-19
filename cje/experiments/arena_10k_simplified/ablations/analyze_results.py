#!/usr/bin/env python3
"""Analyze ablation results and generate summary tables.

This script reads the results from all ablations and creates:
1. Summary statistics table
2. Key findings
3. Figures if requested
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all results from a directory."""
    results = []
    results_file = results_dir / "results.jsonl"

    if results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except:
                    pass

    return results


def analyze_oracle_coverage(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze oracle coverage ablation results."""

    rows = []
    for r in results:
        if r.get("success", False):
            spec = r["spec"]
            rows.append(
                {
                    "Oracle Coverage": f"{spec['oracle_coverage']:.0%}",
                    "RMSE": r.get("rmse_vs_oracle", np.nan),
                    "CI Width": r.get("mean_ci_width", np.nan),
                    "Calibration RMSE": r.get("calibration_rmse", np.nan),
                    "Runtime (s)": r.get("runtime_s", np.nan),
                    "Seed": r.get("seed", 0),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregate by coverage level
    summary = (
        df.groupby("Oracle Coverage")
        .agg(
            {
                "RMSE": ["mean", "std"],
                "CI Width": "mean",
                "Calibration RMSE": "mean",
                "Runtime (s)": "mean",
            }
        )
        .round(4)
    )

    return summary


def analyze_sample_size(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze sample size ablation results."""

    rows = []
    for r in results:
        if r.get("success", False):
            spec = r["spec"]

            # Get mean ESS across policies
            mean_ess = np.nan
            if "ess_absolute" in r:
                mean_ess = np.mean(list(r["ess_absolute"].values()))

            rows.append(
                {
                    "Estimator": spec["estimator"],
                    "Sample Size": spec.get("sample_size", r.get("n_samples", 0)),
                    "RMSE": r.get("rmse_vs_oracle", np.nan),
                    "Mean SE": (
                        np.mean(list(r.get("standard_errors", {}).values()))
                        if r.get("standard_errors")
                        else np.nan
                    ),
                    "ESS": mean_ess,
                    "ESS %": (
                        100 * mean_ess / spec.get("sample_size", 1)
                        if not np.isnan(mean_ess)
                        else np.nan
                    ),
                    "Runtime (s)": r.get("runtime_s", np.nan),
                    "Seed": r.get("seed", 0),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregate by estimator and sample size
    summary = (
        df.groupby(["Estimator", "Sample Size"])
        .agg(
            {
                "RMSE": ["mean", "std"],
                "Mean SE": "mean",
                "ESS": "mean",
                "ESS %": "mean",
                "Runtime (s)": "mean",
            }
        )
        .round(4)
    )

    return summary


def analyze_interaction(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze interaction ablation results."""

    rows = []
    for r in results:
        if r.get("success", False):
            spec = r["spec"]
            rows.append(
                {
                    "Oracle Coverage": f"{spec['oracle_coverage']:.0%}",
                    "Sample Size": spec.get("sample_size", r.get("n_samples", 0)),
                    "N Oracle": int(
                        spec["oracle_coverage"] * spec.get("sample_size", 0)
                    ),
                    "RMSE": r.get("rmse_vs_oracle", np.nan),
                    "CI Width": r.get("mean_ci_width", np.nan),
                    "Runtime (s)": r.get("runtime_s", np.nan),
                    "Seed": r.get("seed", 0),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Create pivot table
    pivot = df.pivot_table(
        values="RMSE", index="Oracle Coverage", columns="Sample Size", aggfunc="mean"
    ).round(4)

    return pivot


def compute_key_findings(
    oracle_df: pd.DataFrame, sample_df: pd.DataFrame, interaction_df: pd.DataFrame
):
    """Extract key findings from results."""

    findings = []

    # Oracle coverage findings
    if not oracle_df.empty:
        rmse_col = ("RMSE", "mean")
        if rmse_col in oracle_df.columns:
            rmse_values = oracle_df[rmse_col].values
            coverages = oracle_df.index

            # Find where RMSE stabilizes (< 5% improvement)
            if len(rmse_values) > 1:
                improvements = -np.diff(rmse_values)
                rel_improvements = improvements / rmse_values[:-1]

                stable_idx = np.where(rel_improvements < 0.05)[0]
                if len(stable_idx) > 0:
                    sweet_spot = coverages[stable_idx[0] + 1]
                    findings.append(
                        f"Oracle coverage sweet spot: {sweet_spot} (diminishing returns beyond)"
                    )

                # Overall improvement from min to max coverage
                total_improvement = (rmse_values[0] - rmse_values[-1]) / rmse_values[0]
                findings.append(
                    f"RMSE reduction from 1% to 100% oracle: {total_improvement:.1%}"
                )

    # Sample size findings
    if not sample_df.empty:
        for estimator in sample_df.index.get_level_values(0).unique():
            est_data = sample_df.loc[estimator]

            if ("RMSE", "mean") in est_data.columns:
                rmse_values = est_data[("RMSE", "mean")].values
                sample_sizes = est_data.index

                if len(rmse_values) >= 2:
                    # Check √n scaling
                    log_n = np.log(sample_sizes)
                    log_rmse = np.log(rmse_values + 1e-10)

                    # Linear regression in log space
                    from scipy import stats

                    slope, _, r_value, _, _ = stats.linregress(log_n, log_rmse)

                    if abs(slope + 0.5) < 0.1:  # Close to -0.5
                        findings.append(
                            f"{estimator}: Follows √n scaling (slope={slope:.2f}, R²={r_value**2:.3f})"
                        )
                    else:
                        findings.append(
                            f"{estimator}: Deviates from √n scaling (slope={slope:.2f})"
                        )

    # Interaction findings
    if not interaction_df.empty:
        # Find most efficient configurations (low RMSE × oracle cost)
        best_configs = []
        for oracle in interaction_df.index:
            for n_samples in interaction_df.columns:
                rmse = interaction_df.loc[oracle, n_samples]
                if not np.isnan(rmse):
                    oracle_pct = float(oracle.strip("%")) / 100
                    n_oracle = oracle_pct * n_samples
                    efficiency = 1.0 / (n_oracle * rmse)
                    best_configs.append((oracle, n_samples, rmse, efficiency))

        if best_configs:
            best_configs.sort(key=lambda x: x[3], reverse=True)
            top = best_configs[0]
            findings.append(
                f"Most efficient config: {top[0]} oracle, n={top[1]} (RMSE={top[2]:.3f})"
            )

    return findings


def generate_summary_table(results_dir: Path = Path("ablations/results")):
    """Generate comprehensive summary table."""

    print("=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Load results
    oracle_results = load_results(results_dir / "oracle_coverage")
    sample_results = load_results(results_dir / "sample_size")
    interaction_results = load_results(results_dir / "interaction")

    print(f"Loaded results:")
    print(f"  Oracle coverage: {len(oracle_results)} experiments")
    print(f"  Sample size: {len(sample_results)} experiments")
    print(f"  Interaction: {len(interaction_results)} experiments")
    print()

    # Analyze each ablation
    oracle_df = analyze_oracle_coverage(oracle_results)
    sample_df = analyze_sample_size(sample_results)
    interaction_df = analyze_interaction(interaction_results)

    # Print oracle coverage results
    if not oracle_df.empty:
        print("ORACLE COVERAGE RESULTS")
        print("-" * 40)
        print(oracle_df)
        print()

    # Print sample size results
    if not sample_df.empty:
        print("SAMPLE SIZE RESULTS")
        print("-" * 40)
        print(sample_df)
        print()

    # Print interaction results
    if not interaction_df.empty:
        print("INTERACTION RESULTS (RMSE)")
        print("-" * 40)
        print(interaction_df)
        print()

    # Compute key findings
    findings = compute_key_findings(oracle_df, sample_df, interaction_df)

    if findings:
        print("KEY FINDINGS")
        print("-" * 40)
        for i, finding in enumerate(findings, 1):
            print(f"{i}. {finding}")
        print()

    # Save to CSV
    if not oracle_df.empty:
        oracle_df.to_csv(results_dir / "oracle_coverage_summary.csv")
        print(f"Saved: {results_dir}/oracle_coverage_summary.csv")

    if not sample_df.empty:
        sample_df.to_csv(results_dir / "sample_size_summary.csv")
        print(f"Saved: {results_dir}/sample_size_summary.csv")

    if not interaction_df.empty:
        interaction_df.to_csv(results_dir / "interaction_summary.csv")
        print(f"Saved: {results_dir}/interaction_summary.csv")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    """Run full analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("ablations/results"),
        help="Directory containing ablation results",
    )
    parser.add_argument("--figures", action="store_true", help="Also generate figures")

    args = parser.parse_args()

    # Generate summary table
    generate_summary_table(args.results_dir)

    # Generate figures if requested
    if args.figures:
        print("\nGenerating figures...")
        # Import and run figure generation from each ablation
        import sys

        sys.path.append("experiments")

        try:
            from oracle_coverage import OracleCoverageAblation
            from sample_size import SampleSizeAblation
            from interaction import InteractionAblation

            # Load and create figures
            oracle_results = load_results(args.results_dir / "oracle_coverage")
            if oracle_results:
                ablation = OracleCoverageAblation()
                ablation.create_figure(
                    oracle_results,
                    args.results_dir
                    / "oracle_coverage"
                    / "figure_1_oracle_coverage.png",
                )
                print("Created: oracle_coverage/figure_1_oracle_coverage.png")

            sample_results = load_results(args.results_dir / "sample_size")
            if sample_results:
                ablation = SampleSizeAblation()
                ablation.create_figure(
                    sample_results,
                    args.results_dir / "sample_size" / "figure_2_sample_scaling.png",
                )
                print("Created: sample_size/figure_2_sample_scaling.png")

            interaction_results = load_results(args.results_dir / "interaction")
            if interaction_results:
                ablation = InteractionAblation()
                ablation.create_figure(
                    interaction_results,
                    args.results_dir / "interaction" / "figure_3_interaction.png",
                )
                print("Created: interaction/figure_3_interaction.png")

        except Exception as e:
            print(f"Error generating figures: {e}")


if __name__ == "__main__":
    main()
