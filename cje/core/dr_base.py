"""Base class for Doubly Robust (DR) estimators.

DR estimators combine a direct method (outcome model) with an IPS correction
to achieve better bias-variance tradeoffs and double robustness properties.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .calibrated_ips import CalibratedIPS
from .outcome_models import IsotonicOutcomeModel, CalibratorBackedOutcomeModel
from ..data.models import EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.fresh_draws import FreshDrawDataset
from ..utils.fresh_draws import validate_fresh_draws

logger = logging.getLogger(__name__)


class DREstimator(CalibratedIPS):
    """Base class for Doubly Robust estimators.

    Key insight: DR = Direct Method + IPS correction
    We inherit from CalibratedIPS to reuse all the IPS machinery,
    and add the direct method (outcome modeling) term.

    The DR formula from the paper (equation 13):
    V_DR(π') = (1/n) Σ [g(X_i, A'_i, S'_i) + W_i * (R_i - g(X_i, A_i, S_i))]

    Where:
    - g is the outcome model (uses cross-fitted isotonic calibration)
    - A'_i are pre-generated fresh draws from the target policy
    - S'_i are pre-evaluated judge scores on fresh draws
    - W_i are the calibrated importance weights (from CalibratedIPS)
    - R_i are the rewards on logged data (from full calibration model)

    Args:
        sampler: PrecomputedSampler with logged data
        outcome_model: Outcome model for predictions (default: IsotonicOutcomeModel)
        n_folds: Number of cross-fitting folds (default 5)
        **kwargs: Additional arguments passed to CalibratedIPS
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        outcome_model: Optional[Any] = None,
        n_folds: int = 5,
        calibrator: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(sampler, **kwargs)

        self.n_folds = n_folds
        self.calibrator = calibrator

        # Choose default outcome model based on available calibrator
        if outcome_model is None:
            if calibrator is not None and hasattr(calibrator, "_fold_models"):
                # We have a cross-fitted calibrator, use it for outcome model
                logger.info(
                    "Using CalibratorBackedOutcomeModel (reusing calibration models)"
                )
                outcome_model = CalibratorBackedOutcomeModel(
                    calibrator, n_folds=n_folds
                )
            else:
                # Check if any samples have cv_fold metadata
                has_cv_fold = any(
                    "cv_fold" in s.metadata
                    for s in sampler.dataset.samples[
                        : min(10, len(sampler.dataset.samples))
                    ]
                )

                if has_cv_fold:
                    logger.warning(
                        "Samples have cv_fold metadata but no calibrator provided. "
                        "Consider passing calibrator from calibrate_dataset() for optimal DR."
                    )

                # Fall back to standard isotonic outcome model
                outcome_model = IsotonicOutcomeModel(n_folds=n_folds)
        self.outcome_model = outcome_model

        # Storage for fresh draws (added via add_fresh_draws)
        self._fresh_draws: Dict[str, FreshDrawDataset] = {}
        self._outcome_fitted = False

        # Generate fold assignments for cross-fitting
        n_samples = len(sampler.dataset.samples)
        self.fold_assignments = self._create_fold_assignments(n_samples, n_folds)

    def _create_fold_assignments(self, n_samples: int, n_folds: int) -> np.ndarray:
        """Create fold assignments for cross-fitting.

        Args:
            n_samples: Number of samples
            n_folds: Number of folds

        Returns:
            Array of fold assignments
        """
        fold_assignments = np.arange(n_samples) % n_folds
        # Shuffle to ensure random assignment
        rng = np.random.RandomState(42)
        rng.shuffle(fold_assignments)
        return fold_assignments

    def add_fresh_draws(self, policy: str, fresh_draws: FreshDrawDataset) -> None:
        """Add pre-generated fresh draws for a target policy.

        Fresh draws must have complete coverage - every logged sample with
        a valid importance weight for this policy must have corresponding
        fresh draws.

        Args:
            policy: Target policy name
            fresh_draws: Pre-generated fresh draw dataset

        Raises:
            ValueError: If fresh draws don't have complete coverage
        """
        # Validate coverage
        validate_fresh_draws(fresh_draws, self.sampler.dataset, policy)

        # Store the fresh draws
        self._fresh_draws[policy] = fresh_draws

        logger.info(
            f"Added fresh draws for policy '{policy}': "
            f"{len(fresh_draws.samples)} samples, "
            f"{fresh_draws.draws_per_prompt} draws/prompt"
        )

    def fit(self) -> None:
        """Fit weight calibration and outcome model."""
        # First fit the IPS weights
        super().fit()

        # Then fit the outcome model on logged data
        self._fit_outcome_model()

        self._fitted = True

    def _fit_outcome_model(self) -> None:
        """Fit the outcome model on logged data."""
        # Collect logged data - only include samples that are valid for at least one policy
        prompts = []
        responses = []
        rewards = []
        judge_scores = []
        valid_fold_assignments = []

        # Get indices of samples that are valid for at least one policy
        valid_for_any = set()
        for policy in self.sampler.target_policies:
            valid_indices = self.sampler._get_valid_indices(policy)
            valid_for_any.update(valid_indices)

        # Sort to maintain order
        valid_indices_list = sorted(valid_for_any)

        for idx in valid_indices_list:
            sample = self.sampler.dataset.samples[idx]
            prompts.append(sample.prompt)
            responses.append(sample.response)

            # Get calibrated reward (from full model)
            if sample.reward is not None:
                rewards.append(sample.reward)
            else:
                raise ValueError("All samples must have calibrated rewards for DR")

            # Get judge score from metadata
            if "judge_score" in sample.metadata:
                judge_scores.append(sample.metadata["judge_score"])
            else:
                raise ValueError("All samples must have judge scores for DR")

            # Get fold assignment - prefer cv_fold from metadata (set by calibration)
            if "cv_fold" in sample.metadata:
                valid_fold_assignments.append(sample.metadata["cv_fold"])
            elif self.fold_assignments is not None:
                valid_fold_assignments.append(self.fold_assignments[idx])

        rewards_array = np.array(rewards)
        judge_scores_array = np.array(judge_scores)
        fold_assignments_array = (
            np.array(valid_fold_assignments) if valid_fold_assignments else None
        )

        # Pass fold assignments to outcome model
        self.outcome_model.fit(
            prompts,
            responses,
            rewards_array,
            judge_scores_array,
            fold_assignments_array,
        )

        # Store the valid indices for later use
        self._outcome_valid_indices = valid_indices_list

        # Precompute prompt_id to fold mapping for O(1) lookup in estimate()
        self._promptid_to_fold = {}
        if fold_assignments_array is not None:
            for idx, fold in zip(valid_indices_list, fold_assignments_array):
                sample = self.sampler.dataset.samples[idx]
                # Require prompt_id for DR
                if "prompt_id" not in sample.metadata:
                    raise ValueError(
                        f"Sample at index {idx} missing 'prompt_id' in metadata. "
                        f"DR estimation requires prompt_id for all samples to align with fresh draws."
                    )
                pid = str(sample.metadata["prompt_id"])
                self._promptid_to_fold[pid] = int(fold)

        self._outcome_fitted = True

        logger.info(f"Fitted outcome model on {len(prompts)} logged samples")

    def estimate(self) -> EstimationResult:
        """Compute DR estimates for all target policies.

        DR formula: V_DR(π') = E[g(X, A', S')] + E[W * (R - g(X, A, S))]
        Where the first term is the Direct Method and second is IPS correction.

        Requires fresh draws to be added via add_fresh_draws() before calling.
        """
        self._validate_fitted()

        estimates = []
        standard_errors = []
        n_samples_used = {}

        for policy in self.sampler.target_policies:
            # Check fresh draws are available
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws for policy '{policy}'. "
                    f"Call add_fresh_draws() before estimate()."
                )

            # Get components
            weights = self.get_weights(policy)
            if weights is None:
                logger.warning(f"No weights for policy '{policy}', skipping")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Get rewards (already filtered to valid samples)
            data = self.sampler.get_data_for_policy(policy)
            if data is None:
                logger.warning(f"No data for policy '{policy}', skipping")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue
            logged_rewards = np.array([d["reward"] for d in data])

            # Sanity check: weights and logged data should be aligned
            if len(weights) != len(logged_rewards):
                raise ValueError(
                    f"Weights and logged data length mismatch for policy '{policy}': "
                    f"weights={len(weights)}, data={len(logged_rewards)}"
                )

            # Get logged data for outcome model
            logged_prompts = [d["prompt"] for d in data]
            logged_responses = [d["response"] for d in data]
            logged_scores = np.array([d.get("judge_score") for d in data])
            # Require prompt_ids for DR (no fallback to index)
            logged_prompt_ids = []
            for i, d in enumerate(data):
                if "prompt_id" not in d:
                    raise ValueError(
                        f"Data entry {i} for policy '{policy}' missing 'prompt_id'. "
                        f"DR estimation requires prompt_id to align with fresh draws."
                    )
                logged_prompt_ids.append(str(d["prompt_id"]))

            # Get fold assignments using precomputed mapping (O(1) lookups)
            valid_fold_ids = []
            if self._promptid_to_fold:
                for pid in logged_prompt_ids:
                    fold = self._promptid_to_fold.get(pid, 0)
                    valid_fold_ids.append(fold)
            valid_fold_ids = np.array(valid_fold_ids)

            # Verify we have the right number of fold assignments
            if len(valid_fold_ids) != len(logged_prompts):
                logger.warning(
                    f"Could not find fold assignments for all samples in policy {policy}: "
                    f"have {len(valid_fold_ids)} fold IDs for {len(logged_prompts)} samples"
                )
                # Fall back to using first fold for missing samples
                while len(valid_fold_ids) < len(logged_prompts):
                    valid_fold_ids = np.append(valid_fold_ids, 0)

            # Get outcome model predictions for logged data (using cross-fitted models)
            # Both IsotonicOutcomeModel and BaseOutcomeModel-derived classes need fold_ids
            if hasattr(self.outcome_model, "predict"):
                # Our outcome models accept fold_ids for cross-fitting
                g_logged = self.outcome_model.predict(
                    logged_prompts, logged_responses, logged_scores, valid_fold_ids
                )
            else:
                # Fallback for other models
                g_logged = self.outcome_model.predict(
                    logged_prompts, logged_responses, logged_scores
                )

            # Get fresh draws
            fresh_dataset = self._fresh_draws[policy]

            # Collect fresh scores for each logged sample
            g_fresh_all = []

            for i, prompt_id in enumerate(logged_prompt_ids):
                # Get fresh judge scores for this prompt
                if prompt_id is None:
                    raise ValueError(f"Missing prompt_id for sample {i}")
                fresh_scores = fresh_dataset.get_scores_for_prompt_id(prompt_id)

                # Get fresh samples to validate fold assignments
                fresh_samples = fresh_dataset.get_samples_for_prompt_id(prompt_id)

                # Create dummy prompts/responses for outcome model interface
                fresh_prompts = [logged_prompts[i]] * len(fresh_scores)
                fresh_responses = [""] * len(fresh_scores)  # Not used in isotonic model

                # Use same fold for all fresh draws from this prompt
                fresh_fold_ids = np.full(len(fresh_scores), valid_fold_ids[i])

                # Validate that fresh draws have matching fold assignments if available
                for j, fresh_sample in enumerate(fresh_samples):
                    if (
                        fresh_sample.fold_id is not None
                        and fresh_sample.fold_id != valid_fold_ids[i]
                    ):
                        logger.warning(
                            f"Fold mismatch for prompt_id '{prompt_id}': "
                            f"logged fold={valid_fold_ids[i]}, fresh fold={fresh_sample.fold_id}"
                        )

                # Get predictions for fresh draws
                # Note: Our models need fold_ids for cross-fitting
                # They all use the same fold for each prompt's fresh draws
                if hasattr(self.outcome_model, "predict"):
                    g_fresh_prompt = self.outcome_model.predict(
                        fresh_prompts, fresh_responses, fresh_scores, fresh_fold_ids
                    )
                else:
                    # Fallback for other models
                    g_fresh_prompt = self.outcome_model.predict(
                        fresh_prompts, fresh_responses, fresh_scores
                    )

                # Average over draws for this prompt
                g_fresh_all.append(g_fresh_prompt.mean())

            g_fresh = np.array(g_fresh_all)

            # Sanity check: weights should have mean approximately 1.0
            weights_mean = weights.mean()
            if not (0.9 <= weights_mean <= 1.1):
                weights_min = weights.min()
                weights_max = weights.max()
                weights_std = weights.std()
                logger.warning(
                    f"Weights for policy '{policy}' deviate from expected mean=1.0: "
                    f"mean={weights_mean:.3f}, std={weights_std:.3f}, "
                    f"min={weights_min:.3e}, max={weights_max:.3e}. "
                    f"This may indicate calibration issues or poor policy overlap."
                )

            # DR estimate components
            dm_term = g_fresh.mean()  # Direct method term
            ips_correction = (weights * (logged_rewards - g_logged)).mean()
            dr_estimate = dm_term + ips_correction

            # Compute standard error using influence function
            if_contributions = (
                g_fresh + weights * (logged_rewards - g_logged) - dr_estimate
            )
            se = np.std(if_contributions, ddof=1) / np.sqrt(len(if_contributions))

            estimates.append(dr_estimate)
            standard_errors.append(se)
            n_samples_used[policy] = len(data)

            logger.info(
                f"DR estimate for policy '{policy}': {dr_estimate:.4f} ± {se:.4f} "
                f"(DM={dm_term:.4f}, IPS_corr={ips_correction:.4f})"
            )

        # Add DR-specific diagnostics
        dr_diagnostics = {
            "fresh_draws_policies": list(self._fresh_draws.keys()),
            "cross_fitted": True,
            "n_folds": self.n_folds,
        }

        # Merge with IPS diagnostics
        all_diagnostics = {**self._diagnostics, **dr_diagnostics}

        return EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="dr_base",
            metadata={
                "diagnostics": all_diagnostics,
                "target_policies": self.sampler.target_policies,
            },
        )


class DRCPOEstimator(DREstimator):
    """DR-CPO: Default DR estimator using isotonic outcome model.

    This is the simplest DR variant that uses g(x,a,s) = f(s) where
    f is the isotonic calibration function learned from judge scores.

    For logged data: Uses cross-fitted predictions f^(-k)(S_i)
    For fresh draws: Uses cross-fitted predictions f^(-k)(S'_i)

    This is theoretically sound under the monotone sufficiency assumption
    (A-J2S) from the paper: E[Y | X, A, S] = μ(S) for some non-decreasing μ.

    By default uses IsotonicOutcomeModel with cross-fitting.
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        outcome_model: Optional[Any] = None,
        n_folds: int = 5,
        calibrator: Optional[Any] = None,
        **kwargs: Any,
    ):
        # Pass everything to parent - it will choose the right outcome model
        super().__init__(
            sampler=sampler,
            outcome_model=outcome_model,
            n_folds=n_folds,
            calibrator=calibrator,
            **kwargs,
        )

    def estimate(self) -> EstimationResult:
        """Override to set correct method name."""
        result = super().estimate()
        result.method = "dr_cpo"
        return result
