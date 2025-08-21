"""Base class for Doubly Robust (DR) estimators.

DR estimators combine a direct method (outcome model) with an IPS correction
to achieve better bias-variance tradeoffs and double robustness properties.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import dataclasses

from .calibrated_ips import CalibratedIPS
from .base_estimator import BaseCJEEstimator
from .outcome_models import IsotonicOutcomeModel, CalibratorBackedOutcomeModel
from ..data.models import EstimationResult
from ..diagnostics import DRDiagnostics, IPSDiagnostics
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.fresh_draws import FreshDrawDataset, validate_fresh_draws
from ..diagnostics.dr import (
    compute_dr_policy_diagnostics,
    compute_orthogonality_score,
    compute_dm_ips_decomposition,
)

logger = logging.getLogger(__name__)


class DREstimator(BaseCJEEstimator):
    """Base class for Doubly Robust estimators with flexible weight method.

    Key insight: DR = Direct Method + IPS correction
    This class uses CalibratedIPS for the importance weighting component,
    which can operate in calibrated or raw mode for flexibility.

    The DR formula from the paper (equation 13):
    V_DR(π') = (1/n) Σ [g(X_i, A'_i, S'_i) + W_i * (R_i - g(X_i, A_i, S_i))]

    Where:
    - g is the outcome model (uses cross-fitted isotonic calibration)
    - A'_i are pre-generated fresh draws from the target policy
    - S'_i are pre-evaluated judge scores on fresh draws
    - W_i are the importance weights (raw or calibrated)
    - R_i are the rewards on logged data (from full calibration model)

    Args:
        sampler: PrecomputedSampler with logged data
        outcome_model: Outcome model for predictions (default: IsotonicOutcomeModel)
        n_folds: Number of cross-fitting folds (default 5)
        use_calibrated_weights: If True, use SIMCal calibration; if False, use raw weights (default True)
        calibrator: Optional calibrator for CalibratorBackedOutcomeModel
        **kwargs: Additional arguments passed to the base class (e.g., oracle_slice_config)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        outcome_model: Optional[Any] = None,
        n_folds: int = 5,
        use_calibrated_weights: bool = True,
        calibrator: Optional[Any] = None,
        random_seed: int = 42,
        run_diagnostics: bool = True,
        **kwargs: Any,
    ):
        # Pass oracle_slice_config to base class (now handles it for all estimators)
        super().__init__(
            sampler=sampler,
            run_diagnostics=run_diagnostics,
            diagnostic_config=None,  # Will use defaults
            **kwargs,  # Passes oracle_slice_config if provided
        )

        self.n_folds = n_folds
        self.calibrator = calibrator
        self.use_calibrated_weights = use_calibrated_weights
        self.random_seed = random_seed

        # Initialize the IPS estimator with appropriate mode
        self.ips_estimator: CalibratedIPS
        # Pass calibrator to CalibratedIPS for DR-aware direction selection if calibrating
        ips_kwargs = {
            "calibrate": use_calibrated_weights,
            "run_diagnostics": run_diagnostics,
        }
        if use_calibrated_weights and calibrator is not None:
            ips_kwargs["calibrator"] = calibrator

        self.ips_estimator = CalibratedIPS(sampler, **ips_kwargs)

        logger.info(
            f"Using CalibratedIPS with calibrate={use_calibrated_weights} for importance weights in DR"
        )

        # IMPORTANT: Share the IPS estimator's augmentation object
        # DR uses IPS weights, so it should use IPS's m(S) fitting
        self.oracle_augmentation = self.ips_estimator.oracle_augmentation

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

        # Store components for diagnostics
        self._dm_component: Dict[str, np.ndarray] = {}
        self._ips_correction: Dict[str, np.ndarray] = {}
        self._fresh_rewards: Dict[str, np.ndarray] = {}
        self._outcome_predictions: Dict[str, np.ndarray] = {}
        self._orthogonality_scores: Dict[str, Dict[str, Any]] = {}
        self._dm_ips_decompositions: Dict[str, Dict[str, Any]] = {}

        # Note: Fold assignments are now computed on-demand from prompt_ids
        # This ensures correct folds even for filtered data

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

    def _compute_policy_diagnostics(
        self, policy: str, estimate: float
    ) -> Dict[str, Any]:
        """Compute diagnostics for a single policy.

        This helper method ensures consistent diagnostic computation across
        all DR estimator subclasses.

        Args:
            policy: Policy name
            estimate: The DR estimate for this policy

        Returns:
            Dictionary of diagnostic metrics
        """
        return compute_dr_policy_diagnostics(
            dm_component=self._dm_component.get(policy, np.array([])),
            ips_correction=self._ips_correction.get(policy, np.array([])),
            dr_estimate=estimate,
            fresh_rewards=self._fresh_rewards.get(policy),  # Always use stored rewards
            outcome_predictions=self._outcome_predictions.get(policy),
            influence_functions=self._influence_functions.get(policy),
            unique_folds=list(range(self.n_folds)),
            policy=policy,
        )

    def fit(self) -> None:
        """Fit weight calibration (if applicable) and outcome model."""
        # First fit the IPS weights
        self.ips_estimator.fit()

        # Then fit the outcome model on logged data
        self._fit_outcome_model()

        self._fitted = True

    def _fit_outcome_model(self) -> None:
        """Fit the outcome model on logged data."""
        # Get indices of samples that are valid for at least one policy
        valid_for_any: set[int] = set()
        for policy in self.sampler.target_policies:
            valid_indices = self.sampler._get_valid_indices(policy)
            valid_for_any.update(valid_indices)

        # Sort to maintain order
        valid_indices_list = sorted(valid_for_any)

        # Upfront validation: Check all samples have judge scores
        missing_judge_scores = []
        invalid_judge_scores = []
        for idx in valid_indices_list:
            sample = self.sampler.dataset.samples[idx]
            if "judge_score" not in sample.metadata:
                missing_judge_scores.append((idx, sample.prompt_id))
            elif sample.metadata["judge_score"] is None:
                invalid_judge_scores.append((idx, sample.prompt_id))
            elif not isinstance(sample.metadata["judge_score"], (int, float)):
                invalid_judge_scores.append((idx, sample.prompt_id))

        if missing_judge_scores:
            example_ids = [str(pid) for _, pid in missing_judge_scores[:3]]
            raise ValueError(
                f"DR requires judge_score for all samples. Missing {len(missing_judge_scores)} scores. "
                f"Example prompt_ids: {example_ids}. "
                f"Run calibrate_dataset(..., enable_cross_fit=True) with judge_field specified."
            )

        if invalid_judge_scores:
            example_ids = [str(pid) for _, pid in invalid_judge_scores[:3]]
            raise ValueError(
                f"DR requires numeric judge_score for all samples. {len(invalid_judge_scores)} invalid. "
                f"Example prompt_ids: {example_ids}."
            )

        # Collect logged data
        prompts = []
        responses = []
        rewards = []
        judge_scores = []
        valid_fold_assignments = []

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

            # Get fold assignment using unified system
            # Note: We compute fold from prompt_id to handle filtered data correctly
            from ..data.folds import get_fold

            fold = get_fold(sample.prompt_id, self.n_folds, self.random_seed)
            valid_fold_assignments.append(fold)

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
                pid = str(sample.prompt_id)
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
            weights = self.ips_estimator.get_weights(policy)
            if weights is None:
                # Check if this is a no_overlap case
                # get_diagnostics() doesn't take policy argument, it returns all
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
            # Strict mode: error if any prompt_id is missing fold assignment
            valid_fold_ids_list = []
            if self._promptid_to_fold:
                unknown_pids = []
                for pid in logged_prompt_ids:
                    if pid not in self._promptid_to_fold:
                        unknown_pids.append(pid)
                    else:
                        valid_fold_ids_list.append(self._promptid_to_fold[pid])

                if unknown_pids:
                    raise ValueError(
                        f"Missing fold assignments for {len(unknown_pids)} samples in policy '{policy}'. "
                        f"Example prompt_ids: {unknown_pids[:3]}. "
                        f"Ensure calibration was done with enable_cross_fit=True or provide explicit fold assignments."
                    )
            else:
                raise ValueError(
                    f"No fold assignments available for DR estimation. "
                    f"Ensure calibration was done with enable_cross_fit=True."
                )
            valid_fold_ids = np.array(valid_fold_ids_list)

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
            fresh_draw_var_per_prompt_list = []  # For diagnostics

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

                # Track variance for diagnostics
                if len(g_fresh_prompt) > 1:
                    fresh_draw_var_per_prompt_list.append(g_fresh_prompt.var())
                else:
                    fresh_draw_var_per_prompt_list.append(0.0)

            g_fresh = np.array(g_fresh_all)
            fresh_draw_var_per_prompt = np.array(fresh_draw_var_per_prompt_list)

            # Sanity check: weights should have mean approximately 1.0
            weights_mean = weights.mean()
            # With mean-one calibration, weights should be very close to 1.0
            # Allow small tolerance for numerical precision
            if not (0.99 <= weights_mean <= 1.01):
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
            ips_correction_base = weights * (logged_rewards - g_logged)

            # Add oracle slice augmentation
            # DR shares the IPS estimator's augmentation (which has m̂(S) fitted)
            # The augmentation corrects for uncertainty in the calibrated rewards f̂(S)
            aug_vector, aug_diagnostics = self.oracle_augmentation.compute_augmentation(
                policy,
                logged_rewards,  # Always use calibrated rewards
                data,
                self.sampler.dataset.samples,
            )
            self._aug_diagnostics[policy] = aug_diagnostics

            # Total IPS correction with augmentation
            ips_correction = (ips_correction_base + aug_vector).mean()
            dr_estimate = dm_term + ips_correction

            # Store components for diagnostics (avoid recomputation later)
            self._dm_component[policy] = g_fresh
            self._ips_correction[policy] = (
                ips_correction_base + aug_vector
            )  # Include augmentation
            self._fresh_rewards[policy] = logged_rewards  # Actually logged rewards
            self._outcome_predictions[policy] = g_logged

            # Compute standard error using influence function
            # Include augmentation in the influence function
            if_contributions = g_fresh + ips_correction_base + aug_vector - dr_estimate
            se = np.std(if_contributions, ddof=1) / np.sqrt(len(if_contributions))

            # Store influence functions (always needed for proper inference)
            self._influence_functions[policy] = if_contributions
            logger.debug(
                f"Stored {len(if_contributions)} influence values for {policy}"
            )

            estimates.append(dr_estimate)
            standard_errors.append(se)
            n_samples_used[policy] = len(data)

            logger.info(
                f"DR estimate for policy '{policy}': {dr_estimate:.4f} ± {se:.4f} "
                f"(DM={dm_term:.4f}, IPS_corr={ips_correction:.4f})"
            )

            # Compute orthogonality score (new)
            ortho_result = compute_orthogonality_score(
                weights=weights,
                rewards=logged_rewards,
                outcome_predictions=g_logged,
                return_ci=True,
            )
            self._orthogonality_scores[policy] = ortho_result

            # Compute DM-IPS decomposition (new)
            decomp_result = compute_dm_ips_decomposition(
                g_hat=g_fresh,
                weights=weights,
                rewards=logged_rewards,
                q_hat=g_logged,
            )
            self._dm_ips_decompositions[policy] = decomp_result

        # Build DR diagnostics using stored components
        dr_diagnostics_per_policy: Dict[str, Dict[str, Any]] = {}

        for idx, policy in enumerate(self.sampler.target_policies):
            if policy not in self._dm_component or np.isnan(estimates[idx]):
                continue

            # Use helper method for consistent diagnostic computation
            dr_diagnostics_per_policy[policy] = self._compute_policy_diagnostics(
                policy, estimates[idx]
            )

        # Collect oracle augmentation diagnostics if available
        oracle_aug_diagnostics: Dict[str, Any] = {}
        if self.use_calibrated_weights and hasattr(
            self.ips_estimator, "_aug_diagnostics"
        ):
            oracle_aug_diagnostics = self.ips_estimator._aug_diagnostics.copy()

        # Add DR-specific metadata
        dr_metadata = {
            "fresh_draws_policies": list(self._fresh_draws.keys()),
            "cross_fitted": True,
            "n_folds": self.n_folds,
            "oracle_slice_augmentation": oracle_aug_diagnostics,  # Add augmentation info
        }

        # Create overview
        dr_overview = {}
        if dr_diagnostics_per_policy:
            dr_overview = {
                "policies": list(dr_diagnostics_per_policy.keys()),
                "dm_vs_ips": {
                    p: (d["dm_mean"], d["ips_corr_mean"])
                    for p, d in dr_diagnostics_per_policy.items()
                },
                "worst_if_tail_ratio_99_5": max(
                    d.get("if_tail_ratio_99_5", 0.0)
                    for d in dr_diagnostics_per_policy.values()
                ),
            }

            # For TMLE specifically (will be overridden in subclass)
            if self.__class__.__name__ == "TMLEEstimator":
                dr_overview["tmle_score_abs_mean"] = {
                    p: abs(d["score_mean"])
                    for p, d in dr_diagnostics_per_policy.items()
                }

        # Build metadata (keep dr_diagnostics for backward compatibility with visualization)
        metadata = {
            "target_policies": list(self.sampler.target_policies),
            "weight_method": "calibrated" if self.use_calibrated_weights else "raw",
            "dr_diagnostics": dr_diagnostics_per_policy,  # Keep for visualization
            "dr_overview": dr_overview,
            "orthogonality_scores": self._orthogonality_scores,  # New: orthogonality diagnostics
            "dm_ips_decompositions": self._dm_ips_decompositions,  # New: DM-IPS breakdown
        }

        # Get IPS diagnostics if available
        ips_diag = None
        if hasattr(self.ips_estimator, "get_diagnostics"):
            ips_diag = self.ips_estimator.get_diagnostics()

        # Build DR diagnostics directly
        dr_diagnostics = self._build_dr_diagnostics(
            estimates,
            standard_errors,
            n_samples_used,
            dr_diagnostics_per_policy,
            ips_diag,
        )

        return EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="dr_base",
            influence_functions=self._influence_functions,
            diagnostics=dr_diagnostics,
            metadata=metadata,
        )

    def _build_dr_diagnostics(
        self,
        estimates: List[float],
        standard_errors: List[float],
        n_samples_used: Dict[str, int],
        dr_diagnostics_per_policy: Dict[str, Dict[str, Any]],
        ips_diagnostics: Optional[IPSDiagnostics],
    ) -> DRDiagnostics:
        """Build DRDiagnostics object from components.

        Args:
            estimates: List of estimates per policy
            standard_errors: List of SEs per policy
            n_samples_used: Dict of samples used per policy
            dr_diagnostics_per_policy: Detailed DR diagnostics
            ips_diagnostics: IPSDiagnostics from internal IPS estimator

        Returns:
            DRDiagnostics object
        """
        # Build estimates/SE dicts
        policies = list(self.sampler.target_policies)
        estimates_dict = {
            p: float(e) for p, e in zip(policies, estimates) if not np.isnan(e)
        }
        se_dict = {
            p: float(se) for p, se in zip(policies, standard_errors) if not np.isnan(se)
        }

        # Extract summary metrics from detailed diagnostics
        r2_values = []
        rmse_values = []
        if_tail_ratios = []

        for policy, diag in dr_diagnostics_per_policy.items():
            if "r2_oof" in diag and diag["r2_oof"] is not None:
                r2_values.append(diag["r2_oof"])
            if "residual_rmse" in diag and diag["residual_rmse"] is not None:
                rmse_values.append(diag["residual_rmse"])
            if "if_tail_ratio_99_5" in diag:
                if_tail_ratios.append(diag["if_tail_ratio_99_5"])
            else:
                # Use a default value if influence functions weren't computed
                if_tail_ratios.append(0.0)

        # Compute ranges
        outcome_r2_range = (min(r2_values), max(r2_values)) if r2_values else (0.0, 0.0)
        outcome_rmse_mean = np.mean(rmse_values) if rmse_values else 0.0
        worst_if_tail = max(if_tail_ratios) if if_tail_ratios else 0.0

        # Build DRDiagnostics
        if ips_diagnostics is not None:
            # Copy fields from IPS diagnostics
            diagnostics = DRDiagnostics(
                estimator_type=f"DR_{ips_diagnostics.estimator_type}",
                method=self.__class__.__name__.lower().replace("estimator", ""),
                n_samples_total=ips_diagnostics.n_samples_total,
                n_samples_valid=ips_diagnostics.n_samples_valid,
                n_policies=len(policies),
                policies=policies,
                estimates=estimates_dict,
                standard_errors=se_dict,
                n_samples_used=n_samples_used,
                # Weight fields from IPS
                weight_ess=ips_diagnostics.weight_ess,
                weight_status=ips_diagnostics.weight_status,
                ess_per_policy=ips_diagnostics.ess_per_policy,
                max_weight_per_policy=ips_diagnostics.max_weight_per_policy,
                weight_tail_ratio_per_policy=ips_diagnostics.weight_tail_ratio_per_policy,
                # Calibration fields (may be None)
                calibration_rmse=ips_diagnostics.calibration_rmse,
                calibration_r2=ips_diagnostics.calibration_r2,
                calibration_coverage=ips_diagnostics.calibration_coverage,
                n_oracle_labels=ips_diagnostics.n_oracle_labels,
                # DR-specific fields
                dr_cross_fitted=True,
                dr_n_folds=self.n_folds,
                outcome_r2_range=outcome_r2_range,
                outcome_rmse_mean=outcome_rmse_mean,
                worst_if_tail_ratio=worst_if_tail,
                dr_diagnostics_per_policy=dr_diagnostics_per_policy,
                dm_ips_decompositions=self._dm_ips_decompositions,
                orthogonality_scores=self._orthogonality_scores,
                influence_functions=self._influence_functions,
            )
        else:
            # No IPS diagnostics available, create minimal version
            from ..diagnostics import Status

            diagnostics = DRDiagnostics(
                estimator_type="DR",
                method=self.__class__.__name__.lower().replace("estimator", ""),
                n_samples_total=len(self.sampler.dataset.samples),
                n_samples_valid=self.sampler.n_valid_samples,
                n_policies=len(policies),
                policies=policies,
                estimates=estimates_dict,
                standard_errors=se_dict,
                n_samples_used=n_samples_used,
                # Minimal weight fields
                weight_ess=0.0,
                weight_status=Status.WARNING,
                ess_per_policy={},
                max_weight_per_policy={},
                weight_tail_ratio_per_policy={},
                # DR-specific fields
                dr_cross_fitted=True,
                dr_n_folds=self.n_folds,
                outcome_r2_range=outcome_r2_range,
                outcome_rmse_mean=outcome_rmse_mean,
                worst_if_tail_ratio=worst_if_tail,
                dr_diagnostics_per_policy=dr_diagnostics_per_policy,
                dm_ips_decompositions=self._dm_ips_decompositions,
                orthogonality_scores=self._orthogonality_scores,
                influence_functions=self._influence_functions,
            )

        return diagnostics

    def get_weights(self, policy: str) -> Optional[np.ndarray]:
        """Get importance weights for a policy.

        Args:
            policy: Target policy name

        Returns:
            Array of importance weights or None if not fitted
        """
        if not self._fitted:
            return None
        return self.ips_estimator.get_weights(policy)

    def get_weight_diagnostics(self) -> Optional[IPSDiagnostics]:
        """Get weight diagnostics from internal IPS estimator.

        This helper method provides easy access to weight diagnostics
        for DR estimators, which internally use an IPS estimator for weights.

        Returns:
            IPSDiagnostics object from the internal IPS estimator, or None
        """
        if hasattr(self.ips_estimator, "get_diagnostics"):
            diag = self.ips_estimator.get_diagnostics()
            # Ensure it's IPSDiagnostics (not some other type)
            if isinstance(diag, IPSDiagnostics):
                return diag
        return None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the estimation.

        Returns:
            Dictionary with diagnostic metrics
        """
        diagnostics: Dict[str, Any] = {
            "weight_method": "calibrated" if self.use_calibrated_weights else "raw",
            "outcome_model": type(self.outcome_model).__name__,
            "n_folds": self.n_folds,
            "policies_with_fresh_draws": list(self._fresh_draws.keys()),
        }

        # Add IPS diagnostics if available
        if hasattr(self.ips_estimator, "get_diagnostics"):
            ips_diag = self.ips_estimator.get_diagnostics()
            # ips_diag is an IPSDiagnostics object, not a dict
            if ips_diag is not None:
                # Convert to dict for legacy compatibility
                diagnostics["ips_weight_ess"] = ips_diag.weight_ess
                diagnostics["ips_n_samples"] = ips_diag.n_samples_valid

        return diagnostics


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
        use_calibrated_weights: bool = True,
        calibrator: Optional[Any] = None,
        random_seed: int = 42,
        **kwargs: Any,
    ):
        # Pass everything to parent - it will choose the right outcome model
        super().__init__(
            sampler=sampler,
            outcome_model=outcome_model,
            n_folds=n_folds,
            use_calibrated_weights=use_calibrated_weights,
            calibrator=calibrator,
            random_seed=random_seed,
            **kwargs,
        )

    def estimate(self) -> EstimationResult:
        """Override to set correct method name."""
        result = super().estimate()
        result.method = "dr_cpo"
        return result
