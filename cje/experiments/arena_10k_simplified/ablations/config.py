"""
Experiment configuration for unified ablation system.

This defines all parameter combinations we want to test.
"""

# Core experiment parameters
EXPERIMENTS = {
    "estimators": [
        "raw-ips",  # Raw IPS (no calibration)
        "calibrated-ips",  # Calibrated IPS
        "orthogonalized-ips",  # Orthogonalized Calibrated IPS
        "dr-cpo",  # DR-CPO
        "oc-dr-cpo",  # Orthogonalized Calibrated DR
        "tr-cpo",  # Triply-Robust CPO (vanilla, raw W)
        "tr-cpo-e",  # Triply-Robust CPO (efficient, m̂(S))
        "stacked-dr",  # Ensemble (always with calibration)
    ],
    "sample_sizes": [500, 1000, 2500, 5000],
    "oracle_coverages": [0.05, 0.10, 0.25, 0.5, 1.00],
    # Key ablation: calibration on/off
    "use_weight_calibration": [
        True,
        False,
    ],  # Test with and without weight calibration (SIMCal)
    # IIC for DR methods
    "use_iic": [True, False],
    # Reward calibration mode (not ablated - just use monotone)
    "reward_calibration_mode": "monotone",
    # Multiple seeds for robust results
    "seeds": [
        42,
        # 123,
        # 456,
        # 789,
        # 1011,
        # 1213,
        # 1415,
        # 1617,
        # 1819,
        # 2021,
    ],  # 10 seeds for statistical robustness
}

# Method-specific constraints
from typing import Dict, Any

# These estimators REQUIRE calibration (can't be turned off)
REQUIRES_CALIBRATION = {
    "calibrated-ips",  # By definition
    "orthogonalized-ips",  # Requires calibrated weights for orthogonalization
    "oc-dr-cpo",  # Orthogonalized Calibrated DR requires calibration
    "stacked-dr",  # Production default - always uses calibration
}

# These estimators can work with or without calibration
CALIBRATION_OPTIONAL = {
    "dr-cpo",  # Can use raw or calibrated weights
}

# These estimators never use weight calibration
NEVER_CALIBRATED = {
    "raw-ips",  # Never uses calibration by design
    "tr-cpo",  # Always uses raw/Hajek weights (no SIMCal) for theoretical correctness
    "tr-cpo-e",  # Also uses raw/Hajek weights, but with m̂(S) in TR term for efficiency
}

CONSTRAINTS = {
    "requires_calibration": REQUIRES_CALIBRATION,
    "calibration_optional": CALIBRATION_OPTIONAL,
    "never_calibrated": NEVER_CALIBRATED,
}

# Fixed parameters for DR methods
DR_CONFIG = {
    "n_folds": 20,  # Increased from 5 to enable reliable cluster-robust inference
    "v_folds_stacking": 20,  # Outer folds for stacked-dr (also increased)
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
