"""Utility functions for diagnostics.

This module contains:
- Weight Diagnostics: Debug importance sampling issues
- Visualization: Plotting utilities for weight diagnostics
"""

from .weight_diagnostics import (
    diagnose_weights,
    create_weight_summary_table,
    detect_api_nondeterminism,
    WeightDiagnostics,
)

# Import visualization functions if matplotlib is available
try:
    from .visualization import (
        plot_weight_distributions,
        plot_ess_comparison,
        plot_weight_summary,
        plot_calibration_comparison,
    )
    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__ = [
    # Weight diagnostics
    "diagnose_weights",
    "create_weight_summary_table",
    "detect_api_nondeterminism",
    "WeightDiagnostics",
]

if _visualization_available:
    __all__.extend([
        # Visualization
        "plot_weight_distributions",
        "plot_ess_comparison", 
        "plot_weight_summary",
        "plot_calibration_comparison",
    ])
