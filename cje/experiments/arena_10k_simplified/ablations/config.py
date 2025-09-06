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
        "tr-cpo",  # Triply-Robust CPO
        "stacked-dr",  # Ensemble
    ],
    "sample_sizes": [500, 1000, 2500, 5000],
    "oracle_coverages": [0.05, 0.10, 0.25, 0.5, 1.00],
    # Key ablation: calibration on/off
    "use_calibration": [True, False],  # Test with and without calibration
    # IIC for DR methods
    "use_iic": [True, False],
    "seed": 42,  # Single seed for simplicity
}

# Method-specific constraints
from typing import Dict, Any

# These estimators REQUIRE calibration (can't be turned off)
REQUIRES_CALIBRATION = {
    "calibrated-ips",  # By definition
    "orthogonalized-ips",  # Requires calibrated weights for orthogonalization
    "oc-dr-cpo",  # Orthogonalized Calibrated DR requires calibration
}

# These estimators can work with or without calibration
CALIBRATION_OPTIONAL = {
    "raw-ips",  # Never uses calibration
    "dr-cpo",  # Can use raw or calibrated weights
    "tr-cpo",  # Uses raw/Hajek weights, but can have calibrated rewards
    "stacked-dr",  # Can use either
}

CONSTRAINTS = {
    "requires_calibration": REQUIRES_CALIBRATION,
    "calibration_optional": CALIBRATION_OPTIONAL,
}

# Fixed parameters for DR methods
DR_CONFIG = {
    "fresh_draws_k": 10,  # Fixed at reasonable value
    "n_folds": 5,
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
