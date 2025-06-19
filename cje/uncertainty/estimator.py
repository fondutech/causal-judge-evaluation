"""Uncertainty-aware causal estimators.

This module provides estimators that natively incorporate judge uncertainty
into confidence intervals and provide detailed variance decomposition.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from .schemas import (
    JudgeScore,
    CalibratedReward,
    UncertaintyAwareEstimate,
    VarianceDecomposition,
)
from .calibration import calibrate_scores_isotonic, apply_full_calibration
from .shrinkage import adaptive_shrinkage, compute_ess, compute_optimal_shrinkage
from .diagnostics import compute_variance_decomposition, analyze_variance_contributions

from ..estimators.base import Estimator
from ..estimators.results import EstimationResult
from .results import MultiPolicyUncertaintyResult, create_multi_policy_result
from ..estimators.reliability import EstimatorMetadata
from sklearn.model_selection import KFold as KFoldSplitter

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimatorConfig:
    """Configuration for uncertainty-aware estimators."""

    # Cross-fitting
    k_folds: int = 5
    random_state: Optional[int] = None

    # Variance shrinkage settings
    use_variance_shrinkage: bool = True
    shrinkage_method: str = "optimal"  # "optimal", "adaptive", "fixed"
    fixed_shrinkage_lambda: float = 1.0
    min_ess_ratio: float = 0.1  # For adaptive shrinkage

    # Calibration settings
    calibrate_variance: bool = True
    min_calibration_range: float = 0.05

    # Diagnostics
    compute_diagnostics: bool = True
    high_variance_percentile: float = 90.0


class UncertaintyAwareDRCPO:
    """Doubly-robust estimator with first-class uncertainty support."""

    def __init__(self, config: UncertaintyEstimatorConfig):
        self.config = config
        self.splitter = KFoldSplitter(
            n_splits=config.k_folds,
            random_state=config.random_state,
            shuffle=True,
        )

    def fit(
        self,
        X: Optional[np.ndarray],
        judge_scores: List[JudgeScore],
        oracle_rewards: np.ndarray,
        importance_weights: np.ndarray,
        policy_names: Optional[List[str]] = None,
    ) -> MultiPolicyUncertaintyResult:
        """Fit uncertainty-aware DR-CPO estimator.

        Args:
            X: Features (not used, for compatibility)
            judge_scores: Judge scores with uncertainty
            oracle_rewards: True rewards for calibration subset
            importance_weights: Importance weights w = π'/π₀
            policy_names: Optional policy names

        Returns:
            EstimationResult with uncertainty-aware confidence intervals
        """
        n_samples = len(judge_scores)
        n_policies = importance_weights.shape[1] if importance_weights.ndim > 1 else 1

        # Ensure 2D weights
        if importance_weights.ndim == 1:
            importance_weights = importance_weights.reshape(-1, 1)

        # Initialize results storage
        uncertainty_estimates = []
        metadata_list = []

        # Process each policy
        for policy_idx in range(n_policies):
            logger.info(f"Processing policy {policy_idx + 1}/{n_policies}")

            # Get weights for this policy
            weights = importance_weights[:, policy_idx]

            # Cross-fitting
            fold_estimates = []
            fold_eif_components = []
            fold_metadata = []

            for fold_idx, (cal_idx, est_idx) in enumerate(
                self.splitter.split(np.arange(n_samples))
            ):
                # Split data
                cal_scores = [judge_scores[i] for i in cal_idx]
                est_scores = [judge_scores[i] for i in est_idx]
                cal_oracle = oracle_rewards[cal_idx]
                est_weights = weights[est_idx]

                # Calibrate on calibration fold
                iso_model, gamma = calibrate_scores_isotonic(cal_scores, cal_oracle)

                # Apply calibration to estimation fold
                calibrated_rewards = apply_full_calibration(
                    est_scores, iso_model, gamma
                )

                # Process this fold
                fold_result = self._process_fold(
                    calibrated_rewards,
                    est_weights,
                    fold_idx,
                    policy_idx,
                )

                fold_estimates.append(fold_result["estimate"])
                fold_eif_components.extend(fold_result["eif_components"])
                fold_metadata.append(fold_result["metadata"])

            # Combine fold results
            estimate = np.mean(fold_estimates)
            eif_components = np.array(fold_eif_components)

            # Compute standard error with uncertainty
            eif_variance = np.var(eif_components)

            # Add judge uncertainty contribution
            all_rewards = []
            all_variances = []
            all_weights = []
            for fold_meta in fold_metadata:
                all_rewards.extend(fold_meta["rewards"])
                all_variances.extend(fold_meta["variances"])
                all_weights.extend(fold_meta["weights"])

            judge_var_contribution = np.mean(
                np.array(all_weights) ** 2 * np.array(all_variances)
            )

            total_variance = eif_variance + judge_var_contribution
            se = np.sqrt(total_variance / n_samples)

            # Create uncertainty-aware estimate
            decomp = compute_variance_decomposition(
                eif_variance, judge_var_contribution
            )

            # Calculate ESS from weights
            weights_array = np.array(all_weights)
            ess = n_samples * np.mean(weights_array) ** 2 / np.mean(weights_array**2)

            uncertainty_estimate = UncertaintyAwareEstimate(
                value=estimate,
                se=se,
                ci_lower=estimate - 1.96 * se,
                ci_upper=estimate + 1.96 * se,
                variance_decomposition=decomp,
                effective_sample_size=ess,
                shrinkage_applied=any(
                    fm.get("shrinkage_applied", False) for fm in fold_metadata
                ),
                shrinkage_lambda=(
                    fold_metadata[0].get("shrinkage_lambda") if fold_metadata else None
                ),
            )

            uncertainty_estimates.append(uncertainty_estimate)

            # Create metadata
            ess = compute_ess(weights)
            metadata = EstimatorMetadata(
                estimator_type="UncertaintyAwareDRCPO",
                k_folds=self.config.k_folds,
                ess_values=[ess],
                ess_percentage=ess / n_samples * 100,
                calibration_range=fold_metadata[0].get("calibration_range", 0),
            )
            metadata_list.append(metadata)

            # Log results
            logger.info(
                f"Policy {policy_idx}: estimate={estimate:.4f}, "
                f"SE={se:.4f}, ESS={ess:.1f}"
            )

        # Create multi-policy result
        return create_multi_policy_result(
            estimates=uncertainty_estimates,
            policy_names=policy_names,
            n_samples=n_samples,
            metadata_list=[m.to_dict() for m in metadata_list],
            estimator_type="UncertaintyAwareDRCPO",
        )

    def _process_fold(
        self,
        calibrated_rewards: List[CalibratedReward],
        weights: np.ndarray,
        fold_idx: int,
        policy_idx: int,
    ) -> Dict[str, Any]:
        """Process a single cross-fitting fold with uncertainty.

        Args:
            calibrated_rewards: Calibrated rewards with variance
            weights: Importance weights for this fold
            fold_idx: Fold index
            policy_idx: Policy index

        Returns:
            Dictionary with fold results
        """
        # Extract values and variances
        rewards = np.array([r.value for r in calibrated_rewards])
        variances = np.array([r.variance for r in calibrated_rewards])

        # Apply variance shrinkage if requested
        if self.config.use_variance_shrinkage:
            # Compute initial estimate for shrinkage
            initial_estimate = np.sum(weights * rewards) / np.sum(weights)

            if self.config.shrinkage_method == "optimal":
                # Use optimal shrinkage
                lambda_star = compute_optimal_shrinkage(
                    weights, rewards, variances, initial_estimate
                )
                shrunk_weights = weights / (1 + lambda_star * variances)

            elif self.config.shrinkage_method == "adaptive":
                # Use adaptive shrinkage with ESS constraint
                shrunk_weights, lambda_used, _ = adaptive_shrinkage(
                    weights,
                    rewards,
                    variances,
                    initial_estimate,
                    min_ess_ratio=self.config.min_ess_ratio,
                )
                lambda_star = lambda_used

            else:  # fixed
                lambda_star = self.config.fixed_shrinkage_lambda
                shrunk_weights = weights / (1 + lambda_star * variances)

            # Use shrunk weights
            final_weights = shrunk_weights
            shrinkage_applied = True

        else:
            # No shrinkage
            final_weights = weights
            shrinkage_applied = False
            lambda_star = None

        # Normalize weights
        final_weights = final_weights / np.mean(final_weights)

        # Compute estimate
        estimate = np.mean(final_weights * rewards)

        # Compute EIF components
        eif_components = final_weights * (rewards - estimate)

        # Collect metadata
        metadata = {
            "rewards": rewards.tolist(),
            "variances": variances.tolist(),
            "weights": final_weights.tolist(),
            "shrinkage_applied": shrinkage_applied,
            "shrinkage_lambda": lambda_star,
            "calibration_range": np.ptp(rewards),
            "ess": compute_ess(final_weights),
        }

        return {
            "estimate": estimate,
            "eif_components": eif_components,
            "metadata": metadata,
        }


def create_uncertainty_aware_result(
    base_result: EstimationResult,
    judge_scores: List[JudgeScore],
    importance_weights: np.ndarray,
    config: UncertaintyEstimatorConfig,
) -> UncertaintyAwareEstimate:
    """Convert standard result to uncertainty-aware result with diagnostics.

    Args:
        base_result: Standard estimation result
        judge_scores: Judge scores with uncertainty
        importance_weights: Importance weights used
        config: Estimator configuration

    Returns:
        UncertaintyAwareEstimate with full diagnostics
    """
    # For simplicity, process first policy
    # In practice, this would be done for each policy
    estimate = base_result.estimates[0]
    se = base_result.se[0]

    # Extract variances
    variances = np.array([s.variance for s in judge_scores])

    # Ensure weights are 1D for first policy
    if importance_weights.ndim > 1:
        weights = importance_weights[:, 0]
    else:
        weights = importance_weights

    # Compute variance decomposition
    eif_variance = base_result.se[0] ** 2 * base_result.n
    judge_variance = np.mean(weights**2 * variances)

    decomp = compute_variance_decomposition(eif_variance, judge_variance)

    # Compute diagnostics if requested
    if config.compute_diagnostics:
        diag = analyze_variance_contributions(weights, variances)
    else:
        diag = None

    # Create uncertainty-aware estimate
    return UncertaintyAwareEstimate(
        value=estimate,
        se=se,
        ci_lower=estimate - 1.96 * se,
        ci_upper=estimate + 1.96 * se,
        variance_decomposition=decomp,
        effective_sample_size=compute_ess(weights),
        shrinkage_applied=config.use_variance_shrinkage,
        shrinkage_lambda=getattr(base_result, "shrinkage_lambda", None),
    )
