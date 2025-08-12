"""Doubly Robust visualization utilities.

DEPRECATED: This module is being phased out.
- plot_dr_dashboard has been moved to dashboards.py
- plot_dr_calibration has been removed (unused)

This file is kept temporarily for backward compatibility.
"""

# Import from new location
from .dashboards import plot_dr_dashboard

# Re-export for backward compatibility
__all__ = ["plot_dr_dashboard"]
