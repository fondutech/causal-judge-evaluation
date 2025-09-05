"""
Experiment configuration for unified ablation system.

This defines all parameter combinations we want to test.
"""

# Core experiment parameters
EXPERIMENTS = {
    "estimators": [
        "ips",  # IPS (can be calibrated or not)
        "dr-cpo",  # Basic DR
        "stacked-dr",  # Ensemble of DR methods
    ],
    "sample_sizes": [500, 1000, 2500, 5000],
    "oracle_coverages": [0.05, 0.10, 0.25, 0.5, 1.00],
    # Key ablation: calibration on/off
    "use_calibration": [True, False],  # Test with and without calibration
    # IIC for DR methods
    "use_iic": [True, False],
    # Reward calibration mode ablation
    "reward_calibration_mode": ["auto", "monotone", "two_stage"],
    # Weight normalization mode ablation (hajek vs raw)
    "weight_mode": ["hajek", "raw"],
    "seed": 42,  # Single seed for simplicity
}

# Method-specific constraints (empty now since all methods test all modes)
from typing import Dict, Any

CONSTRAINTS: Dict[str, Any] = {}

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
    "max_workers": 8,
}
