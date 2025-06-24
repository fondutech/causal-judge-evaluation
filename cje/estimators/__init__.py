"""
Causal estimators and evaluation models.

Performance Note:
DR-CPO and MRDR estimators default to samples_per_policy=2 for good variance reduction
while remaining computationally efficient. Set samples_per_policy=0 for maximum speedup
with no loss of unbiasedness, or increase to 5-10 for maximum variance reduction.
"""

from __future__ import annotations
from typing import Dict, Type, Any, cast

# Base classes
from .base import Estimator

# IPS-only estimators (no outcome modeling)
from .ips_only_estimators import (
    IPS,
    SNIPS,
    CalibratedIPS,
    MultiIPSEstimator,
    MultiSNIPSEstimator,
    CalibratedIPSEstimator,
)

# Doubly-robust estimators (with outcome modeling)
from .doubly_robust_estimators import (
    MultiDRCPOEstimator,
    MultiMRDREstimator,
    DRCPOEstimator,
    MRDREstimator,
)


# Supporting modules
from .featurizer import Featurizer, BasicFeaturizer, SentenceEmbeddingFeaturizer
from . import auto_outcome

# For dynamic class loading
import importlib

# Registry of available estimators (simplified names)
_ESTIMATORS: Dict[str, Type[Estimator[Any]]] = {
    "IPS": MultiIPSEstimator,
    "SNIPS": MultiSNIPSEstimator,
    "CalibratedIPS": CalibratedIPSEstimator,
    "DRCPO": MultiDRCPOEstimator,
    "MRDR": MultiMRDREstimator,
}


def get_class_from_string(class_path_str: str) -> Type[Estimator[Any]]:
    """
    Dynamically load a class from a string path.

    Args:
        class_path_str: String like "module.submodule.ClassName"

    Returns:
        The class object
    """
    module_path, class_name = class_path_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return cast(Type[Estimator[Any]], getattr(module, class_name))


def get_estimator(name: str, **kwargs: Any) -> Estimator[Any]:
    """
    Factory function to create estimator instances.

    Args:
        name: Name of the estimator. Available options:
              - "IPS": Inverse Propensity Scoring
              - "SNIPS": Self-Normalized IPS
              - "CalibratedIPS": IPS with propensity calibration and clipping
              - "DRCPO": Doubly-Robust Cross-Policy Optimization
              - "MRDR": Multi-Robust Doubly-Robust
        **kwargs: Additional arguments passed to the estimator constructor

    Returns:
        Configured estimator instance

    Raises:
        ValueError: If estimator name is not recognized
    """
    if name in _ESTIMATORS:
        estimator_cls = _ESTIMATORS[name]
        return estimator_cls(**kwargs)
    else:
        # Try dynamic loading for custom estimators
        try:
            estimator_cls = get_class_from_string(name)
            return estimator_cls(**kwargs)
        except (ImportError, AttributeError, ValueError):
            available = list(_ESTIMATORS.keys())
            raise ValueError(
                f"Unknown estimator '{name}'. Available estimators: {available}"
            )


__all__ = [
    # Base
    "Estimator",
    "get_estimator",
    # IPS-only
    "IPS",
    "SNIPS",
    "CalibratedIPS",
    "MultiIPSEstimator",
    "MultiSNIPSEstimator",
    "CalibratedIPSEstimator",
    # Doubly-robust
    "MultiDRCPOEstimator",
    "MultiMRDREstimator",
    "DRCPOEstimator",
    "MRDREstimator",
    # Supporting
    "Featurizer",
    "BasicFeaturizer",
    "SentenceEmbeddingFeaturizer",
    "auto_outcome",
]
