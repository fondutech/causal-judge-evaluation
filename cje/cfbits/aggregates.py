"""Aggregation utilities for CF-bits paper tables.

This module provides functions to aggregate CF-bits metrics across experiments
for paper-ready tables and analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def aggregate_cfbits_for_paper(
    results: List[Dict[str, Any]],
    group_by: List[str] = None,
) -> pd.DataFrame:
    """Aggregate CF-bits results for paper tables.

    Args:
        results: List of experiment results with CF-bits metrics
        group_by: Columns to group by (default: estimator, sample_size, oracle_coverage)

    Returns:
        DataFrame with aggregated CF-bits metrics
    """
    if group_by is None:
        group_by = ["estimator", "sample_size", "oracle_coverage"]

    # Extract CF-bits data from results
    rows = []
    for result in results:
        if not result.get("success"):
            continue

        spec = result.get("spec", {})
        cfbits_summary = result.get("cfbits_summary", {})

        for policy, metrics in cfbits_summary.items():
            if not metrics:
                continue

            row = {
                "estimator": spec.get("estimator"),
                "sample_size": spec.get("sample_size"),
                "oracle_coverage": spec.get("oracle_coverage"),
                "policy": policy,
                "seed": spec.get("seed_base", 42),
            }

            # Add CF-bits metrics
            row.update(
                {
                    "bits_tot": metrics.get("bits_tot"),
                    "w_tot": metrics.get("w_tot"),
                    "wid": metrics.get("wid"),
                    "wvar": metrics.get("wvar"),
                    "ifr_oua": metrics.get("ifr_oua"),
                    "aess_oua": metrics.get("aess_oua"),
                    "aessf_lcb": metrics.get("aessf_lcb"),
                    "gate_state": metrics.get("gate_state", "UNKNOWN"),
                }
            )

            rows.append(row)

    if not rows:
        logger.warning("No CF-bits data found in results")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Aggregate by groups
    aggregated = []
    for name, group in df.groupby(group_by):
        agg_row = dict(zip(group_by, name))

        # Compute aggregates
        agg_row.update(
            {
                "n_runs": len(group),
                "bits_tot_mean": group["bits_tot"].mean(),
                "bits_tot_std": group["bits_tot"].std(),
                "w_tot_mean": group["w_tot"].mean(),
                "wid_mean": group["wid"].mean() if "wid" in group else None,
                "wvar_mean": group["wvar"].mean() if "wvar" in group else None,
                "ifr_oua_gmean": geometric_mean(group["ifr_oua"].dropna()),
                "aess_oua_mean": (
                    group["aess_oua"].mean() if "aess_oua" in group else None
                ),
                "aessf_lcb_mean": (
                    group["aessf_lcb"].mean() if "aessf_lcb" in group else None
                ),
            }
        )

        # Gate state distribution
        gate_counts = group["gate_state"].value_counts()
        total_gates = len(group)
        agg_row.update(
            {
                "gate_good_pct": 100 * gate_counts.get("GOOD", 0) / total_gates,
                "gate_warning_pct": 100 * gate_counts.get("WARNING", 0) / total_gates,
                "gate_critical_pct": 100 * gate_counts.get("CRITICAL", 0) / total_gates,
                "gate_refuse_pct": 100 * gate_counts.get("REFUSE", 0) / total_gates,
            }
        )

        # Dominance analysis (wid vs wvar)
        if "wid" in group and "wvar" in group:
            wid_dominance = (group["wid"] > group["wvar"]).mean()
            agg_row["wid_dominance_pct"] = 100 * wid_dominance
        else:
            agg_row["wid_dominance_pct"] = None

        aggregated.append(agg_row)

    return pd.DataFrame(aggregated)


def geometric_mean(values: pd.Series) -> float:
    """Compute geometric mean, handling NaN and zero values."""
    clean_values = values.dropna()
    if len(clean_values) == 0:
        return float(np.nan)
    # Handle zeros by adding small epsilon
    clean_values = clean_values.clip(lower=1e-10)
    return float(np.exp(np.mean(np.log(clean_values))))


def compute_budget_recommendations(
    cfbits_data: pd.DataFrame,
    target_improvement: float = 0.5,  # Target CF-bits improvement
) -> pd.DataFrame:
    """Compute budget recommendations for achieving target CF-bits improvement.

    Args:
        cfbits_data: DataFrame with CF-bits metrics
        target_improvement: Target improvement in CF-bits

    Returns:
        DataFrame with budget recommendations
    """
    recommendations = []

    for _, row in cfbits_data.iterrows():
        rec = {
            "estimator": row.get("estimator"),
            "sample_size": row.get("sample_size"),
            "oracle_coverage": row.get("oracle_coverage"),
            "current_bits": row.get("bits_tot_mean"),
            "target_bits": row.get("bits_tot_mean", 0) + target_improvement,
        }

        # Compute required improvements
        if row.get("ifr_oua_gmean"):
            # Required sample size increase for target bits
            # To gain Δ bits, need to multiply adjusted sample size by 2^(2Δ)
            # Since adjusted sample size = n × IFR, and assuming IFR stays roughly constant,
            # we need to multiply n by 2^(2Δ)
            rec["required_logs_factor"] = 2 ** (2 * target_improvement)

        # Oracle budget recommendations
        if row.get("wid_mean") is not None and row.get("wvar_mean") is not None:
            # If Wid dominates, need more oracle labels
            if row["wid_mean"] > row["wvar_mean"]:
                rec["bottleneck"] = "identification"
                rec["recommendation"] = "Increase oracle labels"
                # Rough estimate: halving Wid requires 4x labels
                rec["oracle_factor_needed"] = 4**target_improvement
            else:
                rec["bottleneck"] = "sampling"
                rec["recommendation"] = "Increase sample size or use DR"
                # For sampling: need n * 2^(2*Δbits)
                rec["sample_factor_needed"] = 2 ** (2 * target_improvement)

        recommendations.append(rec)

    return pd.DataFrame(recommendations)


def format_latex_table(
    df: pd.DataFrame,
    columns: List[str] = None,
    caption: str = "CF-bits Metrics by Estimator",
    label: str = "tab:cfbits",
) -> str:
    """Format DataFrame as LaTeX table for paper.

    Args:
        df: DataFrame with aggregated metrics
        columns: Columns to include (default: key metrics)
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    if columns is None:
        columns = [
            "estimator",
            "sample_size",
            "oracle_coverage",
            "bits_tot_mean",
            "wid_mean",
            "wvar_mean",
            "ifr_oua_gmean",
            "aessf_lcb_mean",
            "gate_good_pct",
        ]

    # Filter to requested columns that exist
    available_cols = [c for c in columns if c in df.columns]
    df_formatted = df[available_cols].copy()

    # Format numeric columns
    format_specs = {
        "bits_tot_mean": "{:.2f}",
        "wid_mean": "{:.3f}",
        "wvar_mean": "{:.3f}",
        "ifr_oua_gmean": "{:.1%}",
        "aessf_lcb_mean": "{:.1%}",
        "gate_good_pct": "{:.0f}%",
        "oracle_coverage": "{:.0%}",
    }

    for col, fmt in format_specs.items():
        if col in df_formatted.columns:
            if "%" in fmt and not col.endswith("_pct"):
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: fmt.format(x) if pd.notna(x) else "-"
                )
            elif col.endswith("_pct"):
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.0f}%" if pd.notna(x) else "-"
                )
            else:
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: fmt.format(x) if pd.notna(x) else "-"
                )

    # Generate LaTeX
    latex = df_formatted.to_latex(
        index=False,
        caption=caption,
        label=label,
        column_format="l" + "r" * (len(available_cols) - 1),
        escape=False,
    )

    # Clean up column names for LaTeX
    replacements = {
        "estimator": "Estimator",
        "sample_size": "n",
        "oracle_coverage": "Oracle %",
        "bits_tot_mean": "CF-bits",
        "wid_mean": "$W_{id}$",
        "wvar_mean": "$W_{var}$",
        "ifr_oua_gmean": "IFR (OUA)",
        "aessf_lcb_mean": "A-ESSF LCB",
        "gate_good_pct": "Good %",
    }

    for old, new in replacements.items():
        latex = latex.replace(old, new)

    return latex


def create_efficiency_leaderboard(
    cfbits_data: pd.DataFrame,
    rmse_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """Create efficiency-aware leaderboard combining RMSE and CF-bits metrics.

    Args:
        cfbits_data: DataFrame with CF-bits metrics
        rmse_data: Optional DataFrame with RMSE results

    Returns:
        Leaderboard DataFrame sorted by combined score
    """
    leaderboard = cfbits_data.copy()

    # Add RMSE if available
    if rmse_data is not None:
        leaderboard = leaderboard.merge(
            rmse_data[["estimator", "sample_size", "oracle_coverage", "rmse"]],
            on=["estimator", "sample_size", "oracle_coverage"],
            how="left",
        )

    # Compute combined score (lower is better)
    # Score = RMSE * width, where width = W0 / 2^bits
    # More bits → smaller width → better score
    if "rmse" in leaderboard.columns:
        # Convert bits to width (assuming W0=1.0 for [0,1] KPIs)
        width = np.power(2.0, -leaderboard["bits_tot_mean"].fillna(0.0))
        leaderboard["width_factor"] = width
        leaderboard["combined_score"] = leaderboard["rmse"] * width
    else:
        # Without RMSE, use negative bits as score (more bits = better = lower score)
        leaderboard["combined_score"] = -leaderboard["bits_tot_mean"].fillna(0.0)

    # Add efficiency rank
    leaderboard["efficiency_rank"] = leaderboard.groupby(
        ["sample_size", "oracle_coverage"]
    )["ifr_oua_gmean"].rank(ascending=False, method="min")

    # Sort by combined score
    leaderboard = leaderboard.sort_values("combined_score")
    leaderboard["overall_rank"] = range(1, len(leaderboard) + 1)

    return leaderboard


def compute_quadrant_reliability(
    cfbits_data: pd.DataFrame,
    size_threshold: int = 2500,
    coverage_threshold: float = 0.25,
) -> pd.DataFrame:
    """Analyze reliability by data regime quadrant.

    Args:
        cfbits_data: DataFrame with CF-bits metrics
        size_threshold: Threshold for small vs large sample size
        coverage_threshold: Threshold for low vs high oracle coverage

    Returns:
        DataFrame with quadrant analysis
    """
    # Define quadrants
    cfbits_data["size_category"] = cfbits_data["sample_size"].apply(
        lambda x: "large" if x >= size_threshold else "small"
    )
    cfbits_data["coverage_category"] = cfbits_data["oracle_coverage"].apply(
        lambda x: "high" if x >= coverage_threshold else "low"
    )
    cfbits_data["quadrant"] = (
        cfbits_data["size_category"]
        + "_n_"
        + cfbits_data["coverage_category"]
        + "_coverage"
    )

    # Analyze by quadrant
    quadrant_analysis = []
    for quadrant, group in cfbits_data.groupby("quadrant"):
        analysis = {
            "quadrant": quadrant,
            "n_estimators": group["estimator"].nunique(),
            "mean_bits": group["bits_tot_mean"].mean(),
            "mean_aessf_lcb": group["aessf_lcb_mean"].mean(),
            "gate_good_pct": group["gate_good_pct"].mean(),
            "gate_refuse_pct": group["gate_refuse_pct"].mean(),
            "wid_dominated_pct": group["wid_dominance_pct"].mean(),
        }
        quadrant_analysis.append(analysis)

    return pd.DataFrame(quadrant_analysis)


def add_provenance(
    df: pd.DataFrame,
    cfbits_version: str = "0.3.0",
    config: Dict[str, Any] = None,
) -> pd.DataFrame:
    """Add provenance information to results DataFrame.

    Args:
        df: DataFrame with results
        cfbits_version: CF-bits module version
        config: Configuration used for computation

    Returns:
        DataFrame with provenance columns added
    """
    df = df.copy()
    df["cfbits_version"] = cfbits_version
    df["timestamp"] = pd.Timestamp.now()

    if config:
        df["alpha"] = config.get("alpha", 0.05)
        df["n_boot"] = config.get("n_boot", 500)
        df["wid_n_bins"] = config.get("wid", {}).get("n_bins", 20)
        df["thresholds_hash"] = hash(str(config.get("thresholds", {})))

    return df
