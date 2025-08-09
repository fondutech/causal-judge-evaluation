"""Utility functions for diagnostics.

This module contains:
- Weight Diagnostics: Debug importance sampling issues
- Visualization: Plotting utilities for weight diagnostics
"""

from .diagnostics import (
    diagnose_weights,
    create_weight_summary_table,
    WeightDiagnostics,
)

from .extreme_weights_analysis import (
    analyze_extreme_weights,
)

# Import visualization functions if matplotlib is available
try:
    from .visualization import (
        plot_weight_dashboard,
        plot_calibration_comparison,
    )

    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__ = [
    # Weight diagnostics
    "diagnose_weights",
    "create_weight_summary_table",
    "WeightDiagnostics",
    # Extreme weights analysis
    "analyze_extreme_weights",
]

if _visualization_available:
    __all__.extend(
        [
            # Visualization
            "plot_weight_dashboard",
            "plot_calibration_comparison",
        ]
    )
