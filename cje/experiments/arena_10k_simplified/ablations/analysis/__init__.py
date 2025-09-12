"""Ablation analysis modules.

Tools for analyzing results from multi-experiment ablation studies.
Each module handles a specific aspect of the analysis.
"""

from .loader import (
    load_results,
    add_ablation_config,
    add_quadrant_classification,
    filter_results,
)

from .rmse import (
    compute_rmse_metrics,
    compute_debiased_rmse,
    aggregate_rmse_by_quadrant,
)

from .coverage import (
    compute_coverage_metrics,
    compute_interval_scores,
    aggregate_coverage_by_estimator,
)

from .bias import (
    compute_bias_analysis,
    compute_bias_by_quadrant,
)

from .diagnostics import (
    compute_diagnostic_metrics,
    compute_boundary_analysis,
    compare_ips_diagnostics,
)


from .ranking import (
    compute_ranking_metrics,
    compute_pairwise_preferences,
    compute_ranking_by_quadrant,
)

from .reports import (
    print_summary_tables,
    print_quadrant_comparison,
    generate_latex_tables,
)

__all__ = [
    # Data loading
    "load_results",
    "add_ablation_config",
    "add_quadrant_classification",
    "filter_results",
    # RMSE analysis
    "compute_rmse_metrics", 
    "compute_debiased_rmse",
    "aggregate_rmse_by_quadrant",
    # Coverage analysis
    "compute_coverage_metrics",
    "compute_interval_scores",
    "aggregate_coverage_by_estimator",
    # Bias analysis
    "compute_bias_analysis",
    "compute_bias_by_quadrant",
    # Diagnostics
    "compute_diagnostic_metrics",
    "compute_boundary_analysis",
    "compare_ips_diagnostics",
    # Ranking
    "compute_ranking_metrics",
    "compute_pairwise_preferences",
    "compute_ranking_by_quadrant",
    # Reporting
    "print_summary_tables",
    "print_quadrant_comparison",
    "generate_latex_tables",
]