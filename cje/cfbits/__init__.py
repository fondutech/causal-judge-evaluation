"""CF-bits: Information accounting for causal inference.

CF-bits provides an information-accounting layer for CJE that decomposes
uncertainty into identification width (structural limits) and sampling width
(statistical noise).
"""

from .core import (
    CFBits,
    compute_cfbits,
    GatesDecision,
    apply_gates,
    logs_for_delta_bits,
    labels_for_oua_reduction,
    fresh_draws_for_dr_improvement,
    bits_to_width,
    width_to_bits,
)
from .sampling import (
    EfficiencyStats,
    SamplingVariance,
    compute_ifr_aess,
    compute_sampling_width,
    compute_estimator_eif,
)
from .overlap import OverlapFloors, estimate_overlap_floors
from .identification import compute_identification_width
from .playbooks import cfbits_report_fresh_draws, cfbits_report_logging_only
from .config import CFBITS_DEFAULTS, GATE_THRESHOLDS, get_config

__all__ = [
    # Core
    "CFBits",
    "compute_cfbits",
    "GatesDecision",
    "apply_gates",
    # Budget helpers
    "logs_for_delta_bits",
    "labels_for_oua_reduction",
    "fresh_draws_for_dr_improvement",
    "bits_to_width",
    "width_to_bits",
    # Sampling
    "EfficiencyStats",
    "SamplingVariance",
    "compute_ifr_aess",
    "compute_sampling_width",
    "compute_estimator_eif",
    # Overlap
    "OverlapFloors",
    "estimate_overlap_floors",
    # Identification
    "compute_identification_width",
    # Playbooks
    "cfbits_report_fresh_draws",
    "cfbits_report_logging_only",
    # Config
    "CFBITS_DEFAULTS",
    "GATE_THRESHOLDS",
    "get_config",
]
