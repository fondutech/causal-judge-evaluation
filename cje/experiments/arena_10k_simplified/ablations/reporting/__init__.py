"""
Information-dense reporting for paper tables.

This module provides functions to generate the core tables (1-3) for the main text
and diagnostic tables (A1-A6) for the appendix, optimized for information density.
"""

from .paper_tables import (
    generate_leaderboard,
    generate_delta_tables,
    generate_stacking_table,
    compute_debiased_rmse,
    compute_interval_score_oa,
    compute_calibration_score,
    compute_se_geomean,
    compute_ranking_metrics,
    compute_paired_deltas,
)

from .appendix_tables import (
    generate_quadrant_leaderboard,
    generate_bias_patterns_table,
    generate_overlap_diagnostics_table,
    generate_oracle_adjustment_table,
    generate_boundary_outlier_table,
    generate_runtime_complexity_table,
    generate_mae_summary_table,
)

__all__ = [
    # Core tables
    "generate_leaderboard",
    "generate_delta_tables",
    "generate_stacking_table",
    # Metrics
    "compute_debiased_rmse",
    "compute_interval_score_oa",
    "compute_calibration_score",
    "compute_se_geomean",
    "compute_ranking_metrics",
    "compute_paired_deltas",
    # Appendix tables
    "generate_quadrant_leaderboard",
    "generate_bias_patterns_table",
    "generate_overlap_diagnostics_table",
    "generate_oracle_adjustment_table",
    "generate_boundary_outlier_table",
    "generate_runtime_complexity_table",
    "generate_mae_summary_table",
]
