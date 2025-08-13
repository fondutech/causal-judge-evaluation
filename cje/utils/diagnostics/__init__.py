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
)

from .dr import (
    compute_dr_policy_diagnostics,
    compute_dr_diagnostics_all,
)

from .display import (
    create_weight_summary_table,
    format_dr_diagnostic_summary,
)

__all__ = [
    # Weight diagnostics
    "effective_sample_size",
    "compute_ess",
    "tail_weight_ratio",
    "mass_concentration",
    "compute_weight_diagnostics",
    # DR diagnostics
    "compute_dr_policy_diagnostics",
    "compute_dr_diagnostics_all",
    # Display utilities
    "create_weight_summary_table",
    "format_dr_diagnostic_summary",
]
