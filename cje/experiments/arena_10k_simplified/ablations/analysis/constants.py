"""Shared constants for ablation analysis modules."""

# Policy names
POLICIES = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]
WELL_BEHAVED_POLICIES = ["clone", "parallel_universe_prompt", "premium"]

# Quadrant definitions
QUADRANT_ORDER = [
    "Small-LowOracle",
    "Small-HighOracle",
    "Large-LowOracle",
    "Large-HighOracle",
]
QUADRANT_ABBREVIATIONS = {
    "Small-LowOracle": "SL",
    "Small-HighOracle": "SH",
    "Large-LowOracle": "LL",
    "Large-HighOracle": "LH",
}

# Statistical constants
DEFAULT_ALPHA = 0.05  # For 95% confidence intervals
Z_CRITICAL_95 = 1.96  # Standard normal critical value for 95% CI

# Oracle ground truth defaults
DEFAULT_N_ORACLE = 4989  # Default number of oracle samples
