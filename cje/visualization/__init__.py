"""Visualization utilities for CJE framework.

This module provides plotting functions for:
- Weight diagnostics and distributions
- DR diagnostics and dashboards
- Calibration plots
"""

from .weight_plots import (
    plot_weight_dashboard,
    plot_calibration_comparison,
    plot_policy_estimates,
)

from .dr_plots import (
    plot_dr_dashboard,
    plot_dr_calibration,
)

__all__ = [
    # Weight visualization
    "plot_weight_dashboard",
    "plot_calibration_comparison",
    "plot_policy_estimates",
    # DR visualization
    "plot_dr_dashboard",
    "plot_dr_calibration",
]
