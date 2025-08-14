"""
Diagnostic computation utilities for CJE.

This package provides functions for computing various diagnostics.
The data structures are in cje.data.diagnostics.
"""

from .weights import (
    effective_sample_size,
    compute_ess,  # Alias for backward compatibility
    tail_weight_ratio,
    mass_concentration,
    compute_weight_diagnostics,
    hill_tail_index,
    hill_tail_index_stable,
)

from .dr import (
    compute_dr_policy_diagnostics,
    compute_dr_diagnostics_all,
    compute_orthogonality_score,
    compute_dm_ips_decomposition,
)

from .display import (
    create_weight_summary_table,
    format_dr_diagnostic_summary,
)

from .stability import (
    kendall_tau_drift,
    sequential_drift_detection,
    reliability_diagram,
    eif_qq_plot_data,
    compute_stability_diagnostics,
)

__all__ = [
    # Weight diagnostics
    "effective_sample_size",
    "compute_ess",
    "tail_weight_ratio",
    "mass_concentration",
    "compute_weight_diagnostics",
    "hill_tail_index",
    "hill_tail_index_stable",
    # DR diagnostics
    "compute_dr_policy_diagnostics",
    "compute_dr_diagnostics_all",
    "compute_orthogonality_score",
    "compute_dm_ips_decomposition",
    # Stability diagnostics
    "kendall_tau_drift",
    "sequential_drift_detection",
    "reliability_diagram",
    "eif_qq_plot_data",
    "compute_stability_diagnostics",
    # Display utilities
    "create_weight_summary_table",
    "format_dr_diagnostic_summary",
]
