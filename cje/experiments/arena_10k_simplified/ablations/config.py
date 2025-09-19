"""
Experiment configuration for unified ablation system.

This defines all parameter combinations we want to test.
"""

import numpy as np

# Core experiment parameters
EXPERIMENTS = {
    "estimators": [
        "raw-ips",  # Raw IPS (no calibration)
        "calibrated-ips",  # Calibrated IPS
        "orthogonalized-ips",  # Orthogonalized Calibrated IPS
        "dr-cpo",  # DR-CPO
        "oc-dr-cpo",  # Orthogonalized Calibrated DR
        "tr-cpo-e",  # Triply-Robust CPO (efficient, m̂(S))
        "tr-cpo-e-anchored-orthogonal",  # TR-CPO (efficient + anchored + orthogonal)
        "stacked-dr",  # Ensemble with dr-cpo, tmle, mrdr (always with calibration)
        "stacked-dr-oc",  # Ensemble with dr-cpo, oc-dr-cpo, tmle, mrdr (adds orthogonalized component)
        "stacked-dr-oc-tr",  # Ensemble with dr-cpo, oc-dr-cpo, tmle, mrdr, tr-cpo-e (adds triply-robust)
    ],
    "sample_sizes": [250, 500, 1000, 2500, 5000],
    "oracle_coverages": [0.05, 0.10, 0.25, 0.5, 1.00],
    # Key ablation: calibration on/off
    "use_weight_calibration": [
        True,
        False,
    ],  # Test with and without weight calibration (SIMCal)
    # Reward calibration mode (not ablated - just use monotone)
    "reward_calibration_mode": "auto",
    # Multiple seeds for robust results
    "seeds": np.arange(0, 50, 1),
    # CF-bits computation (single toggle, not a grid dimension)
    # Set to True to enable CF-bits metrics for all experiments
    "compute_cfbits": True,  # Default ON to gather CF-bits metrics
    # Variance budget (rho) for SIMCal - fixed at 1.0 (doesn't bind in practice)
    # Controls maximum allowed variance: Var(W_calibrated) ≤ var_cap * Var(W_baseline)
    "var_cap": 1.0,  # Fixed at no variance increase (empirically doesn't bind)
}

# Method-specific constraints
from typing import Dict, Any

# These estimators REQUIRE calibration (can't be turned off)
REQUIRES_CALIBRATION = {
    "calibrated-ips",  # By definition
    "orthogonalized-ips",  # Requires calibrated weights for orthogonalization
    "oc-dr-cpo",  # Orthogonalized Calibrated DR requires calibration
    "stacked-dr",  # Production default - always uses calibration
    "stacked-dr-oc",  # Stacked variant with oc-dr-cpo component - always uses calibration
    "stacked-dr-oc-tr",  # Stacked variant with oc-dr-cpo and tr-cpo-e - always uses calibration
}

# These estimators can work with or without calibration
CALIBRATION_OPTIONAL = {
    "dr-cpo",  # Can use raw or calibrated weights
}

# These estimators never use weight calibration
NEVER_CALIBRATED = {
    "raw-ips",  # Never uses calibration by design
    "tr-cpo-e",  # Also uses raw/Hajek weights, but with m̂(S) in TR term for efficiency
    "tr-cpo-e-anchored-orthogonal",  # Uses raw weights with SIMCal anchoring + orthogonalization
}

CONSTRAINTS = {
    "requires_calibration": REQUIRES_CALIBRATION,
    "calibration_optional": CALIBRATION_OPTIONAL,
    "never_calibrated": NEVER_CALIBRATED,
}

# Fixed parameters for DR methods
DR_CONFIG = {
    "n_folds": 5,  # Standard k-fold cross-fitting (faster, still reliable)
    "v_folds_stacking": 5,  # Outer folds for stacked-dr
}

# CF-bits configuration
CFBITS_CONFIG = {
    "n_boot": 500,  # Reduced from default 800 for performance
    "alpha": 0.05,  # 95% confidence intervals
    "random_state": 42,  # Reproducibility
    "compute_tail_index": False,  # Expensive, disabled by default
}

# Paths (absolute to avoid confusion)
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "data" / "cje_dataset.jsonl"
RESULTS_PATH = BASE_DIR / "results" / "all_experiments.jsonl"
CHECKPOINT_PATH = BASE_DIR / "results" / "checkpoint.jsonl"

# Runtime configuration
RUNTIME = {
    "checkpoint_every": 10,  # Save progress every N experiments
    "verbose": True,
    "parallel": False,
    "max_workers": 10,
}
