"""
Experiment configuration for unified ablation system.

This defines all parameter combinations we want to test.
"""

# Core experiment parameters
EXPERIMENTS = {
    "estimators": [
        "raw-ips",  # Baseline: no calibration
        "calibrated-ips",  # IPS with SIMCal
        "dr-cpo",  # Basic DR
        "tmle",  # TMLE
        "mrdr",  # Multiply robust DR
        "stacked-dr",  # Ensemble of DR methods
    ],
    "sample_sizes": [500, 1000, 2500, 5000],
    "oracle_coverages": [0.05, 0.10, 0.25, 0.5, 1.00],
    # Key ablation: calibration on/off
    "use_calibration": [True, False],  # Test with and without calibration
    # IIC for DR methods
    "use_iic": [True, False],
    "n_seeds": 3,
    "seed_base": 42,
}

# Method-specific constraints
CONSTRAINTS = {
    # raw-ips never uses calibration (it's the uncalibrated baseline)
    "raw-ips": {"use_calibration": [False]},
    # calibrated-ips always uses calibration (it's the whole point)
    "calibrated-ips": {"use_calibration": [True]},
    # IIC only applies to DR methods
    "non_dr_methods": ["raw-ips", "calibrated-ips"],
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
    "parallel": True,
    "max_workers": 10,
}
