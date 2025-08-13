"""Visualization utilities for CJE framework.

This module provides plotting functions organized by domain:
- Weight diagnostics and dashboards
- DR diagnostics and dashboards
- Calibration comparison plots
- Policy estimate visualizations
"""

# Import core visualization functions
from .calibration import plot_calibration_comparison
from .estimates import plot_policy_estimates

# Import weight dashboards (with backward-compatible aliases)
from .weight_dashboards import (
    plot_weight_dashboard,  # Alias for plot_weight_dashboard_summary
    plot_weight_dashboard_summary,
    plot_weight_dashboard_per_policy,  # Alias for plot_weight_dashboard_detailed
    plot_weight_dashboard_detailed,
)

# Import DR dashboards
from .dr_dashboards import plot_dr_dashboard

__all__ = [
    # Calibration
    "plot_calibration_comparison",
    # Policy estimates
    "plot_policy_estimates",
    # Weight dashboards
    "plot_weight_dashboard",
    "plot_weight_dashboard_summary",
    "plot_weight_dashboard_per_policy",
    "plot_weight_dashboard_detailed",
    # DR dashboards
    "plot_dr_dashboard",
]
