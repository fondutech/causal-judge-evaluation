"""Utility functions for diagnostics.

This module contains:
- Weight Diagnostics: Debug importance sampling issues
- Visualization: Plotting utilities for weight diagnostics
"""

from .diagnostics.display import (
    create_weight_summary_table,
)

from .extreme_weights_analysis import (
    analyze_extreme_weights,
)

# Import visualization functions if matplotlib is available
try:
    from .visualization import (
        plot_weight_dashboard,
        plot_calibration_comparison,
        plot_policy_estimates,
    )

    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__ = [
    # Weight diagnostics
    "create_weight_summary_table",
    # Extreme weights analysis
    "analyze_extreme_weights",
]

if _visualization_available:
    __all__.extend(
        [
            # Visualization
            "plot_weight_dashboard",
            "plot_calibration_comparison",
            "plot_policy_estimates",
        ]
    )
