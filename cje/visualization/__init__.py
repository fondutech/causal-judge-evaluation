"""Visualization utilities for CJE framework.

This module provides plotting functions for:
- Weight diagnostics and distributions
- DR diagnostics and dashboards
- Calibration plots
- Policy estimates
"""

# Import from new modular structure
from .calibration import plot_calibration_comparison
from .estimates import plot_policy_estimates
from .dashboards import (
    plot_weight_dashboard,
    plot_weight_dashboard_per_policy,
    plot_dr_dashboard,
)
from .combined_dashboard import plot_combined_weight_dashboard

__all__ = [
    # Calibration
    "plot_calibration_comparison",
    # Policy estimates
    "plot_policy_estimates",
    # Dashboards
    "plot_weight_dashboard",
    "plot_weight_dashboard_per_policy",
    "plot_dr_dashboard",
    "plot_combined_weight_dashboard",
]
