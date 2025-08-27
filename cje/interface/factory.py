"""Estimator registry and builder utilities.

Centralizes creation of estimators to avoid drift between CLI choices,
Hydra configs, and analysis code.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import logging

from ..data.precomputed_sampler import PrecomputedSampler
from ..estimators.calibrated_ips import CalibratedIPS
from ..estimators.dr_base import DRCPOEstimator
from ..estimators.mrdr import MRDREstimator
from ..estimators.tmle import TMLEEstimator
from ..estimators.stacking import StackedDREstimator

logger = logging.getLogger(__name__)

# Type alias for builder functions
BuilderFn = Callable[
    [PrecomputedSampler, Dict[str, Any], Optional[Any], bool],
    Union[CalibratedIPS, DRCPOEstimator, MRDREstimator, TMLEEstimator, StackedDREstimator],
]


def _build_calibrated_ips(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> CalibratedIPS:
    # Pass calibrator for DR-aware direction selection if available
    cfg = dict(config)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        cfg.setdefault("calibrator", calibration_result.calibrator)
        if verbose:
            logger.info("Using calibrator for DR-aware SIMCal direction selection")
    return CalibratedIPS(sampler, **cfg)


def _build_raw_ips(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> CalibratedIPS:
    clip_weight = config.get("clip_weight", 100.0)
    return CalibratedIPS(sampler, calibrate=False, clip_weight=clip_weight)


def _build_dr_cpo(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> DRCPOEstimator:
    n_folds = config.get("n_folds", 5)
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for DR outcome model")
        return DRCPOEstimator(sampler, n_folds=n_folds, calibrator=calibration_result.calibrator)
    return DRCPOEstimator(sampler, n_folds=n_folds)


def _build_mrdr(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> MRDREstimator:
    n_folds = config.get("n_folds", 5)
    omega_mode = config.get("omega_mode", "snips")
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for MRDR")
        return MRDREstimator(
            sampler,
            n_folds=n_folds,
            omega_mode=omega_mode,
            calibrator=calibration_result.calibrator,
        )
    return MRDREstimator(sampler, n_folds=n_folds, omega_mode=omega_mode)


def _build_tmle(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> TMLEEstimator:
    n_folds = config.get("n_folds", 5)
    link = config.get("link", "logit")
    if calibration_result and getattr(calibration_result, "calibrator", None):
        if verbose:
            logger.info("Using calibration models for TMLE")
        return TMLEEstimator(
            sampler,
            n_folds=n_folds,
            link=link,
            calibrator=calibration_result.calibrator,
        )
    return TMLEEstimator(sampler, n_folds=n_folds, link=link)


def _build_stacked_dr(
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> StackedDREstimator:
    estimators = config.get("estimators", ["dr-cpo", "tmle", "mrdr"])
    use_outer_split = config.get("use_outer_split", True)
    parallel = config.get("parallel", True)
    if verbose:
        logger.info(f"Using stacked DR with estimators: {estimators}")
    return StackedDREstimator(
        sampler,
        estimators=estimators,
        use_outer_split=use_outer_split,
        parallel=parallel,
    )


REGISTRY: Dict[str, BuilderFn] = {
    "calibrated-ips": _build_calibrated_ips,
    "raw-ips": _build_raw_ips,
    "dr-cpo": _build_dr_cpo,
    "mrdr": _build_mrdr,
    "tmle": _build_tmle,
    "stacked-dr": _build_stacked_dr,
}


def get_estimator_names() -> Tuple[str, ...]:
    return tuple(REGISTRY.keys())


def create_estimator(
    name: str,
    sampler: PrecomputedSampler,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
):
    if name not in REGISTRY:
        raise ValueError(f"Unknown estimator type: {name}")
    return REGISTRY[name](sampler, config, calibration_result, verbose)

