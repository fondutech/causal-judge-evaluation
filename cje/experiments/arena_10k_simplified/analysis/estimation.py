"""Estimator creation and configuration for CJE analysis.

This module handles creating the appropriate estimator (IPS, DR, TMLE, etc.)
based on configuration and adding fresh draws for DR-based estimators.

Following CLAUDE.md: Do one thing well - this module only handles estimation setup.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from cje import (
    CalibratedIPS,
    RawIPS,
    DRCPOEstimator,
    MRDREstimator,
    TMLEEstimator,
    MRDRTMLEEstimator,
    PrecomputedSampler,
    calibrate_dataset,
)
from cje.utils.fresh_draws import load_fresh_draws_auto
import sys
from pathlib import Path

# Add parent directory to path for validation import
sys.path.insert(0, str(Path(__file__).parent.parent))
from validation import validate_no_unnecessary_calibration


def create_estimator(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any] = None,
) -> Union[CalibratedIPS, RawIPS, DRCPOEstimator, MRDREstimator, TMLEEstimator]:
    """Create the appropriate estimator based on args.

    Args:
        args: Command-line arguments
        sampler: PrecomputedSampler with data
        calibrated_dataset: Dataset with rewards
        cal_result: Optional calibration result

    Returns:
        Configured estimator instance
    """
    estimator_config = args.estimator_config or {}

    if args.estimator == "calibrated-ips":
        return CalibratedIPS(sampler)

    elif args.estimator == "raw-ips":
        clip_weight = estimator_config.get("clip_weight", 100.0)
        return RawIPS(sampler, clip_weight=clip_weight)

    elif args.estimator == "dr-cpo":
        return _create_dr_cpo(args, sampler, cal_result, estimator_config)

    elif args.estimator == "mrdr":
        return _create_mrdr(
            args, sampler, calibrated_dataset, cal_result, estimator_config
        )

    elif args.estimator == "tmle":
        return _create_tmle(
            args, sampler, calibrated_dataset, cal_result, estimator_config
        )

    elif args.estimator == "mrdr-tmle":
        return _create_mrdr_tmle(
            args, sampler, calibrated_dataset, cal_result, estimator_config
        )

    else:
        raise ValueError(f"Unknown estimator: {args.estimator}")


def _create_dr_cpo(
    args: Any,
    sampler: PrecomputedSampler,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> DRCPOEstimator:
    """Create DR-CPO estimator."""
    n_folds = estimator_config.get("n_folds", 5)

    if cal_result and cal_result.calibrator:
        dr_estimator = DRCPOEstimator(
            sampler,
            n_folds=n_folds,
            calibrator=cal_result.calibrator,
        )
        print("   Using CalibratorBackedOutcomeModel (reusing calibration models)")
    else:
        dr_estimator = DRCPOEstimator(sampler, n_folds=n_folds)
        print("   Using IsotonicOutcomeModel (refitting models)")

    # Load fresh draws
    print("   Loading fresh draws for DR estimation...")
    add_fresh_draws(dr_estimator, args, sampler, estimator_config)

    return dr_estimator


def _create_mrdr(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> MRDREstimator:
    """Create MRDR estimator."""
    n_folds = estimator_config.get("n_folds", 5)
    omega_mode = estimator_config.get("omega_mode", "snips")

    # MRDR works best with cross-fitted calibration
    if args.oracle_coverage < 1.0 and (not cal_result or not cal_result.calibrator):
        print("   ⚠️  MRDR works best with cross-fitted calibration. Re-calibrating...")
        validate_no_unnecessary_calibration(
            calibrated_dataset, args.oracle_coverage, cal_result
        )
        calibrated_dataset, cal_result = calibrate_dataset(
            calibrated_dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=n_folds,
        )
        sampler = PrecomputedSampler(calibrated_dataset)

    mrdr_estimator = MRDREstimator(sampler, n_folds=n_folds, omega_mode=omega_mode)
    print(f"   Using MRDR with omega_mode='{omega_mode}'")

    # Load fresh draws
    print("   Loading fresh draws for MRDR estimation...")
    add_fresh_draws(mrdr_estimator, args, sampler, estimator_config)

    return mrdr_estimator


def _create_tmle(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> TMLEEstimator:
    """Create TMLE estimator."""
    n_folds = estimator_config.get("n_folds", 5)
    link = estimator_config.get("link", "logit")

    # TMLE works best with cross-fitted calibration
    if args.oracle_coverage < 1.0 and (not cal_result or not cal_result.calibrator):
        print("   ⚠️  TMLE works best with cross-fitted calibration. Re-calibrating...")
        validate_no_unnecessary_calibration(
            calibrated_dataset, args.oracle_coverage, cal_result
        )
        calibrated_dataset, cal_result = calibrate_dataset(
            calibrated_dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=n_folds,
        )
        sampler = PrecomputedSampler(calibrated_dataset)

    tmle_estimator = TMLEEstimator(sampler, n_folds=n_folds, link=link)
    print(f"   Using TMLE with link='{link}'")

    # Load fresh draws
    print("   Loading fresh draws for TMLE estimation...")
    add_fresh_draws(tmle_estimator, args, sampler, estimator_config)

    return tmle_estimator


def _create_mrdr_tmle(
    args: Any,
    sampler: PrecomputedSampler,
    calibrated_dataset: Any,
    cal_result: Optional[Any],
    estimator_config: Dict[str, Any],
) -> MRDRTMLEEstimator:
    """Create MRDR-TMLE estimator."""
    n_folds = estimator_config.get("n_folds", 5)
    omega_mode = estimator_config.get("omega_mode", "w")
    link = estimator_config.get("link", "logit")

    # MRDR-TMLE benefits from cross-fitted calibration
    if args.oracle_coverage < 1.0 and (not cal_result or not cal_result.calibrator):
        print(
            "   ⚠️  MRDR-TMLE works best with cross-fitted calibration. Re-calibrating..."
        )
        validate_no_unnecessary_calibration(
            calibrated_dataset, args.oracle_coverage, cal_result
        )
        calibrated_dataset, cal_result = calibrate_dataset(
            calibrated_dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=n_folds,
        )
        sampler = PrecomputedSampler(calibrated_dataset)

    mrdr_tmle_estimator = MRDRTMLEEstimator(
        sampler,
        n_folds=n_folds,
        omega_mode=omega_mode,
        link=link,
        calibrator=cal_result.calibrator if cal_result else None,
    )
    print(f"   Using MRDR-TMLE with omega_mode='{omega_mode}', link='{link}'")

    # Load fresh draws
    print("   Loading fresh draws for MRDR-TMLE estimation...")
    add_fresh_draws(mrdr_tmle_estimator, args, sampler, estimator_config)

    return mrdr_tmle_estimator


def add_fresh_draws(
    estimator: Any,
    args: Any,
    sampler: PrecomputedSampler,
    estimator_config: Dict[str, Any],
) -> None:
    """Add fresh draws to a DR estimator for all target policies.

    This loads fresh draws from files. It will fail if fresh draws
    are not available - no synthetic fallback.

    Args:
        estimator: DR estimator instance
        args: Command-line arguments (contains data path)
        sampler: PrecomputedSampler with target policies
        estimator_config: Estimator configuration (unused but kept for compatibility)

    Raises:
        FileNotFoundError: If fresh draws are not available
    """
    data_dir = Path(args.data).parent

    for policy in sampler.target_policies:
        # Load fresh draws - will raise FileNotFoundError if missing
        try:
            fresh_draws = load_fresh_draws_auto(
                data_dir=data_dir,
                policy=policy,
                verbose=False,
            )
            # Add to estimator
            estimator.add_fresh_draws(policy, fresh_draws)
            # Print status - we know these are real draws now
            print(f"     ✓ Loaded {len(fresh_draws.samples)} fresh draws for {policy}")
        except FileNotFoundError as e:
            # Enhance the error message with specific guidance
            raise FileNotFoundError(
                f"No fresh draws found for policy '{policy}' in {data_dir}.\n"
                f"DR/MRDR/TMLE require real fresh draws from teacher forcing.\n"
                f"Options:\n"
                f"  1. Generate fresh draws using generate_fresh_draws.py\n"
                f"  2. Use --estimator calibrated-ips or raw-ips (no fresh draws needed)\n"
                f"  3. Ensure response files exist in {data_dir}/responses/\n"
                f"Original error: {e}"
            ) from e
