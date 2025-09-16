"""Configuration and defaults for CF-bits.

This module centralizes all configuration parameters, thresholds, and defaults
for the CF-bits information accounting system.
"""

from typing import Dict, Any


# Main CF-bits configuration defaults
CFBITS_DEFAULTS: Dict[str, Any] = {
    # Significance level for confidence intervals
    "alpha": 0.05,
    # Bootstrap settings for overlap floors
    "n_boot": 500,  # Reduced from 800 for performance in ablations
    # Tail index computation (expensive, off by default)
    "compute_tail_index": False,
    # Random state for reproducibility
    "random_state": 42,
    # Identification width (Wid) settings
    "wid": {
        "n_bins": 20,  # Number of bins for isotonic bands
        "min_labels_per_bin": 3,  # Minimum oracle labels per bin
        "ci_boot": 0,  # Bootstrap samples for Wid CI (0 = fast default)
    },
}


# Gate thresholds for reliability decisions
GATE_THRESHOLDS: Dict[str, float] = {
    # Structural overlap on σ(S) - A-ESSF thresholds
    "aessf_refuse": 0.05,  # < 5%: Catastrophic overlap, refuse evaluation
    "aessf_critical": 0.20,  # < 20%: Poor overlap, strongly recommend DR
    # Information Fraction Ratio (efficiency)
    "ifr_critical": 0.20,  # < 20%: Very inefficient estimator
    "ifr_warning": 0.50,  # < 50%: Inefficient, consider improvements
    # Tail index for weight distribution
    "tail_critical": 2.0,  # < 2.0: Infinite variance risk
    "tail_warning": 2.5,  # < 2.5: Heavy tails, be cautious
    # Oracle variance dominance
    "oracle_warning": 1.0,  # > 1.0: Oracle uncertainty dominates
    # Identification width (absolute on [0,1] KPI scale)
    "wid_warning": 0.50,  # > 0.5: Identification uncertainty is large
    "wid_critical": 0.80,  # > 0.8: Identification dominates total width
}


# Maximum width for catastrophic cases
# For [0,1] KPIs, refuse if Wmax > 1.0 (worse than no information)
WMAX_THRESHOLD = 1.0  # Catastrophic overlap threshold on [0,1] scale


# Default configuration for ablations
ABLATIONS_CFBITS_CONFIG: Dict[str, Any] = {
    "n_boot": 500,  # Reduced for performance
    "alpha": 0.05,  # 95% confidence intervals
    "random_state": 42,  # Reproducibility
    "compute_tail_index": False,  # Expensive, disabled by default
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged dictionary with overrides applied
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override the value
            result[key] = value

    return result


def get_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get CF-bits configuration with optional custom overrides.

    Args:
        custom_config: Optional custom configuration to merge with defaults

    Returns:
        Complete configuration dictionary
    """
    config = CFBITS_DEFAULTS.copy()
    config["thresholds"] = GATE_THRESHOLDS.copy()

    if custom_config:
        config = deep_merge(config, custom_config)

    return config


def get_ablations_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get CF-bits configuration optimized for ablations.

    Args:
        custom_config: Optional custom configuration to merge with ablations defaults

    Returns:
        Configuration dictionary optimized for ablations
    """
    config = ABLATIONS_CFBITS_CONFIG.copy()

    if custom_config:
        config = deep_merge(config, custom_config)

    return config
