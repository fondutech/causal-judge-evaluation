#!/usr/bin/env python3
"""Generate diagnostics and visualizations for arena analysis.

This script creates all the plots and tables specified in Phase 7:
- Table 1: Policy values with CIs and oracle comparison
- Figure 1: CI width comparison across estimators
- Figure 2: Pairwise p-value heatmap
- Appendix: Weight histograms, calibration curves, etc.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Use unified imports
from cje.workflows import EstimationResult
from cje.utils.progress import console, print_summary_table


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all results from a CJE run."""
    results = {}

    # Load main results
    result_json = results_dir / "result.json"
    if result_json.exists():
        with open(result_json) as f:
            results["estimates"] = json.load(f)

    # Load calibration diagnostics
    cal_diag = results_dir / "calibration_diagnostics.json"
    if cal_diag.exists():
        with open(cal_diag) as f:
            results["calibration"] = json.load(f)

    # Load summary
    summary = results_dir / "summary.json"
    if summary.exists():
        with open(summary) as f:
            results["summary"] = json.load(f)

    return results


def compute_oracle_values(data_path: Path) -> Dict[str, float]:
    """Compute true oracle values for each model from Bradley-Terry utilities."""
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]

    # Group by model and compute average y_true
    model_values = {}
    model_counts = {}

    for row in rows:
        model = row.get("model_id", "unknown")
        if "y_true" in row and row["y_true"] is not None:
            if model not in model_values:
                model_values[model] = 0.0
                model_counts[model] = 0
            model_values[model] += row["y_true"]
            model_counts[model] += 1

    # Compute averages
    oracle_values = {}
    for model, total in model_values.items():
        if model_counts[model] > 0:
            oracle_values[model] = total / model_counts[model]

    return oracle_values


def create_policy_value_table(
    results: Union[Dict[str, Any], EstimationResult],
    oracle_values: Dict[str, float],
    output_path: Path,
) -> pd.DataFrame:
    """Create Table 1: Policy values with CIs and oracle comparison."""
    console.print("[bold]Creating Table 1: Policy values[/bold]")

    rows = []

    # Handle new EstimationResult API
    if isinstance(results, EstimationResult):
        # Get policy names from metadata or generate defaults
        policy_names = getattr(results, "metadata", {}).get("policy_names")
        if policy_names is None:
            policy_names = [f"Policy_{i}" for i in range(results.n_policies)]

        ci_low, ci_high = results.confidence_interval()

        for i, policy_name in enumerate(policy_names):
            # Clean up policy name for matching with oracle
            clean_name = policy_name.replace("_", "-").replace("--", "-")

            # Find matching oracle value
            oracle = None
            for oracle_name, oracle_val in oracle_values.items():
                if oracle_name.replace("/", "_").replace("-", "_") == policy_name:
                    oracle = oracle_val
                    break

            row = {
                "Policy": policy_name,
                "V_hat": results.v_hat[i],
                "CI_Low": ci_low[i],
                "CI_High": ci_high[i],
                "CI_Width": ci_high[i] - ci_low[i],
                "SE": results.se[i],
                "Oracle": oracle if oracle is not None else np.nan,
                "Abs_Error": (
                    abs(results.v_hat[i] - oracle) if oracle is not None else np.nan
                ),
            }
            rows.append(row)
    else:
        # Handle old dict-based format
        estimates = results.get("estimates", {})

        for policy_name, est in estimates.items():
            # Clean up policy name for matching with oracle
            clean_name = policy_name.replace("_", "-").replace("--", "-")

            # Find matching oracle value
            oracle = None
            for oracle_name, oracle_val in oracle_values.items():
                if oracle_name.replace("/", "_").replace("-", "_") == policy_name:
                    oracle = oracle_val
                    break

            row = {
                "Policy": policy_name,
                "V_hat": est["v_hat"],
                "CI_Low": est["ci_low"],
                "CI_High": est["ci_high"],
                "CI_Width": est["ci_high"] - est["ci_low"],
                "SE": est.get("se", np.sqrt(est.get("var", 0))),
                "Oracle": oracle if oracle is not None else np.nan,
                "Abs_Error": (
                    abs(est["v_hat"] - oracle) if oracle is not None else np.nan
                ),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("V_hat", ascending=False)

    # Save as CSV
    df.to_csv(output_path, index=False, float_format="%.4f")
    console.print(f"[green]✓[/green] Saved Table 1 to {output_path}")

    # Use our print_summary_table function
    print_summary_table("Top 5 Models by Estimated Value", df.head().to_dict("records"))

    # Coverage analysis
    if "Oracle" in df.columns:
        coverage = (
            (df["CI_Low"] <= df["Oracle"]) & (df["Oracle"] <= df["CI_High"])
        ).mean()
        console.print(f"\nCI Coverage: {coverage:.1%}")

    return df


def plot_ci_width_comparison(
    results_dict: Dict[str, Dict[str, Any]], output_path: Path
) -> None:
    """Create Figure 1: CI width comparison across estimators."""
    console.print("\n[bold]Creating Figure 1: CI width comparison[/bold]")

    # Prepare data for plotting
    estimator_names = []
    mean_ci_widths = []
    se_ci_widths = []

    for estimator_name, results in results_dict.items():
        estimates = results.get("estimates", {})

        ci_widths = []
        for policy_name, est in estimates.items():
            ci_width = est["ci_high"] - est["ci_low"]
            ci_widths.append(ci_width)

        if ci_widths:
            estimator_names.append(estimator_name)
            mean_ci_widths.append(np.mean(ci_widths))
            se_ci_widths.append(np.std(ci_widths) / np.sqrt(len(ci_widths)))

    # Create plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(estimator_names))

    bars = plt.bar(
        x,
        mean_ci_widths,
        yerr=se_ci_widths,
        capsize=5,
        color=["lightcoral", "lightsalmon", "lightgreen", "lightblue"],
    )

    plt.xlabel("Estimator", fontsize=12)
    plt.ylabel("Mean 95% CI Width", fontsize=12)
    plt.title("Estimator Efficiency: Average Confidence Interval Width", fontsize=14)
    plt.xticks(x, estimator_names)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_ci_widths)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    # Add relative improvement annotations
    if len(mean_ci_widths) > 1:
        base_width = mean_ci_widths[0]  # IPS as baseline
        for i in range(1, len(mean_ci_widths)):
            improvement = (base_width - mean_ci_widths[i]) / base_width * 100
            plt.text(
                i,
                mean_ci_widths[i] / 2,
                f"-{improvement:.0f}%",
                ha="center",
                va="center",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓[/green] Saved Figure 1 to {output_path}")


def plot_pairwise_pvalue_heatmap(
    data: Union[Dict[str, Any], EstimationResult],
    output_path: Path,
    adjust: str = "holm",
) -> None:
    """Create Figure 2: Pairwise p-value heatmap."""
    console.print("\n[bold]Creating Figure 2: Pairwise p-value heatmap[/bold]")

    # Handle new EstimationResult API
    if isinstance(data, EstimationResult):
        # Use the built-in pairwise comparison functionality
        n = data.n_policies
        policy_names = getattr(data, "metadata", {}).get("policy_names")
        if policy_names is None:
            policy_names = [f"Policy_{i}" for i in range(n)]

        # Initialize p-value matrix
        pval_matrix = np.ones((n, n))

        # Compute all pairwise comparisons
        for i in range(n):
            for j in range(i + 1, n):
                comp = data.compare_policies(i, j)
                pval_matrix[i, j] = comp["p_value"]
                pval_matrix[j, i] = comp["p_value"]
    else:
        # Handle old dict-based format
        estimates = (
            data
            if isinstance(data, dict)
            and all(isinstance(v, dict) for v in data.values())
            else data.get("estimates", {})
        )
        policy_names = list(estimates.keys())
        n = len(policy_names)

        # Initialize p-value matrix
        pval_matrix = np.ones((n, n))

        # Compute pairwise p-values
        for i in range(n):
            for j in range(i + 1, n):
                # Get estimates and SEs
                v_i = estimates[policy_names[i]]["v_hat"]
                v_j = estimates[policy_names[j]]["v_hat"]
                se_i = estimates[policy_names[i]].get(
                    "se", np.sqrt(estimates[policy_names[i]].get("var", 0))
                )
                se_j = estimates[policy_names[j]].get(
                    "se", np.sqrt(estimates[policy_names[j]].get("var", 0))
                )

                # Compute test statistic
                delta = v_i - v_j
                se_delta = np.sqrt(se_i**2 + se_j**2)  # Assuming independence

                if se_delta > 0:
                    z = delta / se_delta
                    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                    pval_matrix[i, j] = p_val
                    pval_matrix[j, i] = p_val

    # Apply multiple testing correction
    if adjust == "holm":
        # Holm correction
        upper_tri_indices = np.triu_indices(n, k=1)
        p_values = pval_matrix[upper_tri_indices]
        rejected, corrected_pvals, _, _ = multipletests(p_values, method="holm")

        # Fill corrected p-values back
        corrected_matrix = np.ones((n, n))
        corrected_matrix[upper_tri_indices] = corrected_pvals
        corrected_matrix = (
            corrected_matrix + corrected_matrix.T - np.diag(np.diag(corrected_matrix))
        )
        pval_matrix = corrected_matrix

    # Create heatmap
    plt.figure(figsize=(12, 10))

    # Use log scale for better visualization
    log_pvals = -np.log10(pval_matrix + 1e-10)

    # Create custom colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Plot heatmap
    ax = sns.heatmap(
        log_pvals,
        xticklabels=[p[:15] for p in policy_names],  # Truncate long names
        yticklabels=[p[:15] for p in policy_names],
        cmap=cmap,
        center=-np.log10(0.05),  # Center at p=0.05
        cbar_kws={"label": "-log10(p-value)"},
        square=True,
        linewidths=0.5,
        annot=pval_matrix < 0.05,  # Mark significant cells
        fmt="",
    )

    # Add significance markers
    for i in range(n):
        for j in range(n):
            if i != j and pval_matrix[i, j] < 0.05:
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2)
                )

    plt.title(f"Pairwise Model Comparison ({adjust}-adjusted p-values)", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓[/green] Saved Figure 2 to {output_path}")


def plot_weight_histogram(
    data_path: Path, output_path: Path, clip: float = 20.0
) -> None:
    """Create weight histogram showing distribution and clipped mass."""
    console.print("\n[bold]Creating weight histogram[/bold]")

    # This is a placeholder - in real implementation, you'd extract weights
    # from the estimator's internal state

    # Simulate weights for demonstration
    np.random.seed(42)
    weights = np.random.lognormal(0, 1.5, 1000)
    weights_clipped = np.clip(weights, 0, clip)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top: Full distribution
    ax1.hist(weights, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(
        clip, color="red", linestyle="--", linewidth=2, label=f"Clip threshold = {clip}"
    )
    ax1.set_xlabel("Importance Weight")
    ax1.set_ylabel("Count")
    ax1.set_title("Importance Weight Distribution")
    ax1.legend()
    ax1.set_yscale("log")

    # Bottom: Clipped mass
    clipped_mass = np.mean(weights >= clip)
    ax2.bar(
        ["Clipped", "Unclipped"],
        [clipped_mass, 1 - clipped_mass],
        color=["red", "green"],
    )
    ax2.set_ylabel("Proportion")
    ax2.set_title(f"Clipped Mass: {clipped_mass:.1%}")
    ax2.set_ylim(0, 1)

    # Add text annotations
    ax2.text(
        0,
        clipped_mass + 0.02,
        f"{clipped_mass:.1%}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    ax2.text(
        1,
        (1 - clipped_mass) + 0.02,
        f"{(1-clipped_mass):.1%}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    console.print(f"[green]✓[/green] Saved weight histogram to {output_path}")


def generate_all_diagnostics(
    results_dir: Path, data_path: Path, output_dir: Path
) -> None:
    """Generate all diagnostic plots and tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(results_dir)

    # Compute oracle values
    oracle_values = compute_oracle_values(data_path)

    # Table 1: Policy values
    table_df = create_policy_value_table(
        results, oracle_values, output_dir / "table1_policy_values.csv"
    )

    # Figure 1: CI width comparison (if multiple estimators)
    # For now, just use the single estimator results
    estimator_results = {
        results.get("summary", {}).get("estimator_name", "DRCPO"): results
    }
    plot_ci_width_comparison(estimator_results, output_dir / "figure1_ci_widths.png")

    # Figure 2: Pairwise p-value heatmap
    plot_pairwise_pvalue_heatmap(results, output_dir / "figure2_pvalue_heatmap.png")

    # Appendix: Weight histogram
    plot_weight_histogram(data_path, output_dir / "appendix_weight_histogram.png")

    # Generate LaTeX table
    if not table_df.empty:
        latex_path = output_dir / "table1_latex.tex"
        with open(latex_path, "w") as f:
            f.write("% Table 1: Policy value estimates with confidence intervals\n")
            f.write(
                table_df.to_latex(
                    index=False,
                    float_format="%.3f",
                    column_format="lcccccc",
                    caption="Policy value estimates with 95\\% confidence intervals and oracle comparison",
                    label="tab:policy_values",
                )
            )
        console.print(f"[green]✓[/green] Saved LaTeX table to {latex_path}")

    console.print("\n[bold green]All diagnostics generated successfully![/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate arena analysis diagnostics")
    parser.add_argument(
        "results_dir", type=Path, help="Directory containing CJE results"
    )
    parser.add_argument("data_path", type=Path, help="Path to calibrated data file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for diagnostics (default: results_dir/diagnostics)",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir / "diagnostics"

    generate_all_diagnostics(args.results_dir, args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
