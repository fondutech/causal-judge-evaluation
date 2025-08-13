"""
High-level analysis functions for CJE.

This module provides simple, one-line analysis functions that handle
the complete CJE workflow automatically.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
import numpy as np

from . import load_dataset_from_jsonl, calibrate_dataset
from .data.models import Dataset, EstimationResult
from .data.precomputed_sampler import PrecomputedSampler
from .estimators.calibrated_ips import CalibratedIPS
from .estimators.raw_ips import RawIPS
from .estimators.dr_base import DRCPOEstimator
from .estimators.mrdr import MRDREstimator
from .estimators.tmle import TMLEEstimator

logger = logging.getLogger(__name__)


def analyze_dataset(
    dataset_path: str,
    estimator: str = "calibrated-ips",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    estimator_config: Optional[Dict[str, Any]] = None,
    fresh_draws_dir: Optional[str] = None,
    verbose: bool = False,
) -> EstimationResult:
    """
    Analyze a CJE dataset with automatic workflow orchestration.

    This high-level function handles:
    - Data loading and validation
    - Automatic reward handling (pre-computed, oracle direct, or calibration)
    - Estimator selection and configuration
    - Fresh draw loading for DR estimators
    - Complete analysis workflow

    Args:
        dataset_path: Path to JSONL dataset file
        estimator: Estimator type ("calibrated-ips", "raw-ips", "dr-cpo", "mrdr", "tmle")
        oracle_coverage: Fraction of oracle labels to use for calibration (0.0-1.0)
        judge_field: Metadata field containing judge scores
        oracle_field: Metadata field containing oracle labels
        estimator_config: Optional configuration dict for the estimator
        fresh_draws_dir: Directory containing fresh draw response files (for DR)
        verbose: Whether to print progress messages

    Returns:
        EstimationResult with estimates, standard errors, and metadata

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset is invalid or estimation fails

    Example:
        >>> # Simple usage
        >>> results = analyze_dataset("my_data.jsonl")
        >>> print(f"Best estimate: {results.estimates.max():.3f}")

        >>> # Advanced usage with DR
        >>> results = analyze_dataset(
        ...     "my_data.jsonl",
        ...     estimator="dr-cpo",
        ...     estimator_config={"n_folds": 10},
        ...     fresh_draws_dir="responses/"
        ... )
    """
    if verbose:
        logger.info(f"Loading dataset from {dataset_path}")

    # Step 1: Load dataset
    dataset = load_dataset_from_jsonl(dataset_path)

    if verbose:
        logger.info(f"Loaded {dataset.n_samples} samples")
        logger.info(f"Target policies: {', '.join(dataset.target_policies)}")

    # Step 2: Handle rewards
    calibrated_dataset, calibration_result = _prepare_rewards(
        dataset, judge_field, oracle_field, verbose
    )

    # Step 3: Create sampler
    sampler = PrecomputedSampler(calibrated_dataset)

    if verbose:
        logger.info(f"Valid samples after filtering: {sampler.n_valid_samples}")

    # Step 4: Create and configure estimator
    estimator_obj = _create_estimator(
        sampler, estimator, estimator_config or {}, calibration_result, verbose
    )

    # Step 5: Add fresh draws for DR estimators
    if estimator in ["dr-cpo", "mrdr", "tmle"]:
        # Type narrowing for mypy
        if isinstance(estimator_obj, (DRCPOEstimator, MRDREstimator, TMLEEstimator)):
            _add_fresh_draws(
                estimator_obj,
                sampler,
                calibrated_dataset,
                fresh_draws_dir,
                estimator_config or {},
                verbose,
            )

    # Step 6: Run estimation
    if verbose:
        logger.info(f"Running {estimator} estimation...")

    results = estimator_obj.fit_and_estimate()

    # Add metadata for downstream use
    results.metadata["dataset_path"] = dataset_path
    results.metadata["estimator"] = estimator
    # Note: oracle_coverage removed - production always uses all available oracle labels
    results.metadata["target_policies"] = list(sampler.target_policies)

    # Add estimator config if provided
    if estimator_config:
        results.metadata["estimator_config"] = estimator_config

    # Add field names for reference
    results.metadata["judge_field"] = judge_field
    results.metadata["oracle_field"] = oracle_field

    if verbose:
        logger.info("Analysis complete!")

    return results


def _prepare_rewards(
    dataset: Dataset,
    judge_field: str,
    oracle_field: str,
    verbose: bool,
) -> tuple[Dataset, Optional[Any]]:
    """Prepare rewards through calibration or use existing."""

    # Check if rewards already exist
    rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)

    if rewards_exist > 0:
        if verbose:
            logger.info(f"Using pre-computed rewards ({rewards_exist} samples)")
        return dataset, None

    # Always calibrate using all available oracle labels
    if verbose:
        logger.info("Calibrating judge scores with oracle labels")

    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field=judge_field,
        oracle_field=oracle_field,
        enable_cross_fit=True,  # Always enable for potential DR use
        n_folds=5,
    )

    if verbose and cal_result:
        logger.info(f"Calibration RMSE: {cal_result.calibration_rmse:.3f}")
        logger.info(f"Coverage (±0.1): {cal_result.coverage_at_01:.1%}")

    return calibrated_dataset, cal_result


def _create_estimator(
    sampler: PrecomputedSampler,
    estimator_type: str,
    config: Dict[str, Any],
    calibration_result: Optional[Any],
    verbose: bool,
) -> Union[CalibratedIPS, RawIPS, DRCPOEstimator, MRDREstimator, TMLEEstimator]:
    """Create the appropriate estimator."""

    if estimator_type == "calibrated-ips":
        return CalibratedIPS(sampler, **config)

    elif estimator_type == "raw-ips":
        clip_weight = config.get("clip_weight", 100.0)
        return RawIPS(sampler, clip_weight=clip_weight)

    elif estimator_type == "dr-cpo":
        n_folds = config.get("n_folds", 5)
        # Use calibrator if available for efficiency
        if calibration_result and calibration_result.calibrator:
            if verbose:
                logger.info("Using calibration models for DR outcome model")
            return DRCPOEstimator(
                sampler, n_folds=n_folds, calibrator=calibration_result.calibrator
            )
        else:
            return DRCPOEstimator(sampler, n_folds=n_folds)

    elif estimator_type == "mrdr":
        n_folds = config.get("n_folds", 5)
        omega_mode = config.get("omega_mode", "snips")
        return MRDREstimator(
            sampler,
            n_folds=n_folds,
            omega_mode=omega_mode,
        )

    elif estimator_type == "tmle":
        n_folds = config.get("n_folds", 5)
        link = config.get("link", "logit")
        return TMLEEstimator(
            sampler,
            n_folds=n_folds,
            link=link,
        )

    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


def _add_fresh_draws(
    estimator: Union[DRCPOEstimator, MRDREstimator, TMLEEstimator],
    sampler: PrecomputedSampler,
    dataset: Dataset,
    fresh_draws_dir: Optional[str],
    config: Dict[str, Any],
    verbose: bool,
) -> None:
    """Add fresh draws to a DR estimator."""
    from .utils.fresh_draws import load_fresh_draws_auto

    for policy in sampler.target_policies:
        if fresh_draws_dir:
            # Load from directory - no fallback
            fresh_draws = load_fresh_draws_auto(
                Path(fresh_draws_dir),
                policy,
                verbose=verbose,
            )
        else:
            # No fresh draws available - fail clearly
            raise ValueError(
                f"DR estimators require fresh draws for policy '{policy}'. "
                f"Please provide --fresh-draws-dir with real teacher forcing responses."
            )

        estimator.add_fresh_draws(policy, fresh_draws)

        if verbose:
            logger.info(f"Added {len(fresh_draws.samples)} fresh draws for {policy}")
