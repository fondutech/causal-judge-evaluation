"""
Doubly-Robust Causal-Preference Off-policy (DR-CPO) estimators.

This module contains multi-policy DR-CPO estimators for the CJE framework's unified architecture.
All estimators handle multiple policies, with single-policy evaluation being treated as the K=1 case.

**Architecture:**
- MultiDRCPOEstimator: Native multi-policy implementation with vectorized outputs
- The CLI automatically uses the unified multi-policy approach
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, cast, Type, Union
import numpy as np
import math
from sklearn.linear_model import Ridge

from cje.utils.progress import ProgressMonitor, console, maybe_track

# Optional import to avoid segfault issues on some systems
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    # Fallback dummy that just raises if used
    class XGBRegressor:  # type: ignore
        """Placeholder class when XGBoost is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("xgboost could not be imported")


from joblib import Parallel, delayed
from .base import Estimator
from .results import EstimationResult
from ..loggers.multi_target_sampler import MultiTargetSampler
from .featurizer import Featurizer, BasicFeaturizer
from .auto_outcome import auto_select
import warnings

# Define a type alias for scikit-learn compatible models
ModelType = Any  # Could be more specific if we have a protocol for fit/predict

# Prefer a lightweight model by default to avoid binary compat issues
# Ridge regression is pure-python/scikit-learn C-ext and rarely segfaults
DEFAULT_OUTCOME_MODEL = Ridge


class MultiDRCPOEstimator(Estimator[Dict[str, Any]]):
    """
    Vectorised DR-CPO for K target policies.

    This estimator extends DR-CPO to evaluate multiple target policies
    {π¹, π², ..., πᴷ} simultaneously, returning vectorized estimates
    and a full sandwich covariance matrix.

    The estimator computes:
    - Point estimates: v_hat (K,)
    - Standard errors: se (K,)
    - Full covariance: Sigma_hat (K, K)

    Args:
        sampler: MultiTargetSampler instance for K target policies (required)
        k: Number of cross-validation folds (default: 5)
        seed: Random seed for shuffling data (default: 0)
        outcome_model_cls: The class of the outcome model to use (default: Ridge)
        outcome_model_kwargs: Keyword arguments for instantiating the outcome model
        featurizer: Featurizer instance for transforming log items (default: BasicFeaturizer)
        n_jobs: Number of parallel jobs for cross-validation (default: -1, uses all processors)
        samples_per_policy: Number of samples to generate for each policy during DR estimation (default: 2)
            Setting to 2 provides good variance reduction. Setting to 0 provides speedup with no bias.
            Increase (e.g., to 5-10) if you need maximum variance reduction.

        stabilize_weights: Whether to apply numerical stabilization for extreme log differences (default: True)
            When True, applies log-space stabilization to prevent numerical overflow/underflow.
            When False, uses raw log probabilities (for research/debugging extreme variance).
        calibrate_weights: Whether to apply isotonic calibration to importance weights (default: True)
            When True, applies per-fold isotonic regression to ensure E[w] = 1 while preserving monotonicity.
            When False, uses raw importance weights (may have poor numerical properties).
        calibrate_outcome: Whether to apply isotonic calibration to outcome model predictions (default: True)
            When True, applies isotonic regression to calibrate outcome model predictions against true rewards.
            When False, uses raw outcome model predictions (may be poorly calibrated).
        score_target_policy_sampled_completions: Whether to score target policy samples with judge (default: True)
            Only applies when samples_per_policy > 0 and judge_runner is provided.
            Set to False to skip scoring generated samples (faster but less accurate outcome model).
            Target samples and scores are automatically logged to target_samples.jsonl for debugging/analysis.
        verbose: Whether to print progress inside _compute_mu_pi_matrix (default: False)
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        k: int = 5,
        seed: int = 0,
        outcome_model_cls: Type[ModelType] = DEFAULT_OUTCOME_MODEL,
        outcome_model_kwargs: Optional[Dict[str, Any]] = None,
        featurizer: Optional[Featurizer] = None,
        n_jobs: Optional[int] = -1,
        samples_per_policy: int = 2,
        stabilize_weights: bool = True,
        calibrate_weights: bool = True,
        calibrate_outcome: bool = True,
        verbose: bool = False,
        judge_runner: Optional[Any] = None,
        score_target_policy_sampled_completions: bool = True,
        work_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(verbose=verbose)
        self.sampler = sampler
        self.k = k
        self.seed = seed
        self.outcome_model_cls = outcome_model_cls
        self.outcome_model_kwargs = outcome_model_kwargs or {}
        self.featurizer = featurizer or BasicFeaturizer()
        self.n_jobs = n_jobs
        self.samples_per_policy = samples_per_policy
        self.stabilize_weights = stabilize_weights
        self.calibrate_weights = calibrate_weights
        self.calibrate_outcome = calibrate_outcome
        self.judge_runner = judge_runner
        self.score_target_policy_sampled_completions = (
            score_target_policy_sampled_completions
        )
        self.work_dir = work_dir

        # Storage for target samples (logged automatically)
        self._target_sample_logs: List[Dict[str, Any]] = []

        # Track whether the user provided custom outcome model / featurizer so we
        # can fall back to sensible defaults automatically. The heuristic is:
        #   • If outcome_model_cls is the library default (Ridge) **and**
        #     no custom kwargs were supplied **and** featurizer is left as None,
        #     we treat this as "auto" mode and choose a suitable combination at
        #     fit-time based on the amount of labeled data.
        # If judge_runner is provided, we also automatically add judge scores to features.
        self._auto_select_models = (
            outcome_model_cls is Ridge
            and not outcome_model_kwargs
            and (featurizer is None or isinstance(featurizer, BasicFeaturizer))
        )

        # Initialize state
        self.n: int = 0
        self.K: int = sampler.K  # Number of target policies
        self._folds: Optional[List[np.ndarray]] = None
        self._full_logs_data: List[Dict[str, Any]] = []
        self._rewards_full: Optional[np.ndarray] = None
        self._features_full: Optional[np.ndarray] = None
        self._logp_behavior_full: Optional[np.ndarray] = None

        # Results storage
        self.W: Optional[np.ndarray] = None  # Importance weights matrix (n, K)
        self.m_hat: Optional[np.ndarray] = None  # Outcome model predictions (n,)
        self.m_pi: Optional[np.ndarray] = None  # Target policy expectations (n, K)
        self.r: Optional[np.ndarray] = None  # Rewards (n,)
        self._weight_stats: Optional[Dict[str, Any]] = None  # Weight statistics

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """
        Fit k-fold cross-validated multi-policy DR estimator.

        Args:
            logs: List of logged data points with required fields:
                  - context: Input context
                  - response: Generated sequence
                  - logp: Log probability under behavior policy
                  - reward: Observed reward
        """
        self._full_logs_data = logs
        self.n = len(logs)

        if self.n == 0:
            raise ValueError("Cannot fit estimator with empty logs")

        # ------------------------------------------------------------------
        # Automatic outcome-model/featurizer selection (shared helper)
        # ------------------------------------------------------------------
        if self._auto_select_models:
            (
                self.outcome_model_cls,
                self.outcome_model_kwargs,
                base_featurizer,
            ) = auto_select(self.n)

            # If judge_runner is provided, wrap with ScoreAugmentFeaturizer to include judge scores
            if self.judge_runner is not None:
                from .auto_outcome import ScoreAugmentFeaturizer

                self.featurizer = ScoreAugmentFeaturizer(base_featurizer)
            else:
                self.featurizer = base_featurizer

        # Extract basic data
        self._rewards_full = np.array([log["reward"] for log in logs], dtype=float)
        self._logp_behavior_full = np.array([log["logp"] for log in logs], dtype=float)

        # Compute importance weights matrix W (n, K)
        console.print(
            f"[bold blue]Computing importance weights for {self.K} policies...[/bold blue]"
        )
        contexts = [log["context"] for log in logs]
        responses = [log["response"] for log in logs]
        self.W, self._weight_stats = self.sampler.importance_weights_matrix(
            contexts,
            responses,
            self._logp_behavior_full.tolist(),
            stabilize=self.stabilize_weights,
            return_stats=True,
        )

        # Note: Calibration will be applied per-fold in _process_fold()
        if self.calibrate_weights:
            console.print(
                "[bold blue]✓ Isotonic weight calibration enabled for DRCPO[/bold blue]"
            )
        if self.calibrate_outcome:
            console.print(
                "[bold blue]✓ Isotonic outcome calibration enabled for DRCPO[/bold blue]"
            )

        # Fit featurizer and transform logs
        console.print("[bold blue]Featurizing data...[/bold blue]")
        self.featurizer.fit(logs)
        self._features_full = self.featurizer.transform(logs)

        # Set up cross-validation folds
        indices = np.arange(self.n)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        self._folds = np.array_split(indices, self.k)

        # Store rewards for easy access
        self.r = self._rewards_full

    def _compute_mu_pi_matrix(
        self, log_items: List[Dict[str, Any]], outcome_model: Any, fold_label: str = ""
    ) -> np.ndarray:
        """
        Compute μ_πᵏ(x) = E_πᵏ[μ(x,s)] for all K policies using live sampling.

        Args:
            log_items: List of log entries
            outcome_model: Fitted outcome model
            fold_label: Label for the current fold

        Returns:
            Matrix of shape (len(log_items), K) with μ_πᵏ(x) values
        """
        n_items = len(log_items)
        mu_pi_matrix = np.zeros((n_items, self.K))

        # Use simple console messages instead of nested progress bar to avoid Rich conflicts
        if self.verbose and n_items > 3:
            console.print(
                f"[dim]{fold_label}: sampling from {self.K} policies for {n_items} items...[/dim]"
            )

        for i, log_item in enumerate(log_items):
            # Show periodic progress for verbose mode
            if self.verbose and n_items > 10 and (i + 1) % max(1, n_items // 4) == 0:
                console.print(
                    f"[dim]  {fold_label}: {i+1}/{n_items} items processed[/dim]"
                )

            context = str(log_item["context"])
            samples_per_policy = self.sampler.sample_many(
                context, n=self.samples_per_policy
            )

            for k in range(self.K):
                if samples_per_policy[k]:
                    # Create mock items for each sample
                    mock_items: List[Dict[str, Any]] = [
                        {"context": context, "response": response}
                        for response in samples_per_policy[k]
                    ]

                    # If judge_runner is available and scoring is enabled, score the samples with the judge
                    if (
                        self.judge_runner is not None
                        and self.score_target_policy_sampled_completions
                        and self.samples_per_policy > 0
                    ):
                        for mock_item in mock_items:
                            try:
                                # Score with judge and add as raw score
                                judge_score = self.judge_runner.score(
                                    mock_item["context"], mock_item["response"]
                                )
                                mock_item["score_raw"] = judge_score
                            except Exception as e:
                                # Fallback to no score if judge fails
                                console.print(
                                    f"[yellow]⚠️  Judge scoring failed: {e}[/yellow]"
                                )
                                mock_item["score_raw"] = 0.0

                    # Log target samples automatically
                    if self.samples_per_policy > 0:
                        # Get the correct policy name from the sampler
                        if hasattr(self.sampler, "policy_names") and k < len(
                            self.sampler.policy_names
                        ):
                            policy_name = self.sampler.policy_names[k]
                        else:
                            policy_name = (
                                f"policy_{k}"  # Fallback for old sampler instances
                            )

                        # Store each sample with complete information
                        for mock_item in mock_items:
                            sample_log = {
                                "context": context,
                                "response": mock_item["response"],
                                "policy_name": policy_name,
                                "policy_index": k,
                                "fold_label": fold_label,
                                "context_index": i,
                            }
                            # Add judge score if available
                            if "score_raw" in mock_item:
                                sample_log["judge_score"] = mock_item["score_raw"]

                            self._target_sample_logs.append(sample_log)

                    feats_s = self.featurizer.transform(mock_items)
                    mu_pi_matrix[i, k] = float(outcome_model.predict(feats_s).mean())
                else:
                    mu_pi_matrix[i, k] = 0.0

        if self.verbose and n_items > 3:
            console.print(
                f"[dim]{fold_label}: completed sampling for {n_items} items[/dim]"
            )

        return mu_pi_matrix

    def _write_target_samples_log(self) -> None:
        """Write logged target samples to disk with all information in one file."""
        if self.samples_per_policy == 0 or not self._target_sample_logs:
            return

        # Use explicit work directory or fall back to current directory
        import os
        from pathlib import Path
        import json

        if self.work_dir:
            work_dir = Path(self.work_dir)
        else:
            # Try to detect from temp file or use current directory
            work_dir = Path.cwd()
            temp_work_file = Path("_last_work_dir.txt")
            if temp_work_file.exists():
                try:
                    work_dir = Path(temp_work_file.read_text().strip())
                except Exception:
                    pass

        try:
            # Write comprehensive target samples log
            samples_file = work_dir / "target_samples.jsonl"
            with open(samples_file, "w") as f:
                for sample_log in self._target_sample_logs:
                    f.write(json.dumps(sample_log) + "\n")

            console.print(f"[green]📝 Target samples logged to: {samples_file}[/green]")

            # Summary statistics
            num_samples = len(self._target_sample_logs)
            num_scored = sum(
                1 for log in self._target_sample_logs if "judge_score" in log
            )
            unique_contexts = len(
                set(log["context"] for log in self._target_sample_logs)
            )
            unique_policies = len(
                set(log["policy_name"] for log in self._target_sample_logs)
            )

            console.print(
                f"[dim]   Samples: {num_samples}, Scored: {num_scored}, Contexts: {unique_contexts}, Policies: {unique_policies}[/dim]"
            )

        except Exception as e:
            console.print(
                f"[yellow]⚠️  Failed to write target samples log: {e}[/yellow]"
            )

    def _process_fold(self, fold_indices: np.ndarray, fold_num: int = 0) -> np.ndarray:
        """
        Process a single cross-validation fold.

        This method is designed to be called in parallel by joblib.

        Args:
            fold_indices: Indices for the test fold
            fold_num: The fold number (1-indexed) for labeling

        Returns:
            EIF components for this fold (n_test, K)
        """
        # Determine train indices (all folds except current)
        assert self._folds is not None  # Type assertion for mypy
        if len(self._folds) == 1:
            # For k=1, use leave-one-out or at least warn
            warnings.warn(
                "k=1 cross-validation uses same data for train/test. "
                "Consider k>=5 for proper cross-fitting to avoid overfitting bias.",
                RuntimeWarning,
            )
            train_indices = fold_indices  # Use all data for both train and test
        else:
            train_indices = np.concatenate(
                [
                    f_idx
                    for f_idx in self._folds
                    if not np.array_equal(f_idx, fold_indices)
                ]
            )
        test_indices = fold_indices

        # Train outcome model on training fold
        mu = self.outcome_model_cls(**self.outcome_model_kwargs)
        assert self._features_full is not None  # Type assertion for mypy
        assert self._rewards_full is not None  # Type assertion for mypy
        X_train = self._features_full[train_indices]
        y_train = self._rewards_full[train_indices]

        if X_train.shape[0] > 0:
            mu.fit(X_train, y_train)

        # Evaluate on test fold
        assert self._full_logs_data is not None  # Type assertion for mypy
        test_log_items = [self._full_logs_data[idx] for idx in test_indices]

        # Compute μ(x,y) for observed (context, response) pairs
        X_test = self._features_full[test_indices]
        mu_hat_test = mu.predict(X_test)  # Shape: (n_test,)

        # Compute μ_πᵏ(x) for all K policies
        mu_pi_test = self._compute_mu_pi_matrix(
            test_log_items, mu, f"Fold {fold_num}"
        )  # Shape: (n_test, K)

        # Get importance weights and rewards for test fold
        assert self.W is not None  # Type assertion for mypy
        W_test = self.W[test_indices]  # Shape: (n_test, K) - Raw weights
        r_test = self._rewards_full[test_indices]  # Shape: (n_test,)

        # Apply isotonic weight calibration per fold if enabled
        if self.calibrate_weights:
            from .calibration import calibrate_weights_isotonic

            # Get fold indices for this test set
            fold_indices = np.zeros(
                len(test_indices)
            )  # All test samples belong to current fold

            # Apply isotonic calibration to weights
            W_test_calibrated, calib_diagnostics = calibrate_weights_isotonic(
                W_test, fold_indices=fold_indices, target_mean=1.0
            )
        else:
            # Use raw weights without calibration
            W_test_calibrated = W_test

        # Apply outcome model calibration if enabled
        if self.calibrate_outcome:
            from .calibration import calibrate_outcome_model_isotonic

            # Calibrate outcome model predictions against training rewards
            train_preds = mu.predict(X_train)
            train_rewards = y_train

            calibration_fn, outcome_diagnostics = calibrate_outcome_model_isotonic(
                train_preds, train_rewards
            )

            # Apply calibration to test predictions
            mu_hat_test_calibrated = calibration_fn(mu_hat_test)
        else:
            mu_hat_test_calibrated = mu_hat_test

        # Compute EIF components: μ_πᵏ(x) + wᵏ * (r - μ(x,y))
        # Broadcasting: (n_test, K) + (n_test, K) * (n_test, 1)
        # With calibration: uses calibrated weights and/or calibrated outcome model
        eif_test = mu_pi_test + W_test_calibrated * (
            r_test[:, np.newaxis] - mu_hat_test_calibrated[:, np.newaxis]
        )

        return cast(np.ndarray, eif_test)

    def estimate(self) -> EstimationResult:
        """
        Return the multi-policy DR-CPO estimates.

        Returns:
            EstimationResult containing:
                v_hat: Point estimates (K,)
                se: Standard errors (K,)
                covariance_matrix: Full covariance (K, K)
                And various helper methods for analysis
        """
        if (
            self._folds is None
            or self.W is None
            or self._rewards_full is None
            or self._features_full is None
        ):
            raise RuntimeError("Must call fit() before estimate()")

        # Show progress for cross-validation
        console.print(
            f"[bold blue]Running {self.k}-fold cross-validation...[/bold blue]"
        )

        # Parallel processing of cross-validation folds
        # Use joblib.Parallel with delayed for embarrassingly parallel computation
        with ProgressMonitor() as progress:
            progress.add_task(
                "cv_folds", f"Processing {self.k} cross-validation folds", total=self.k
            )

            all_eif_components: List[np.ndarray] = []

            # Use sequential processing for small datasets to avoid pickling issues
            if self.n < 100 or self.n_jobs == 1 or self.k <= 2:
                console.print(
                    f"[yellow]Using sequential processing (n={self.n}, k={self.k}, n_jobs={self.n_jobs})[/yellow]"
                )
                results: List[np.ndarray] = []

                for i, fold_indices in enumerate(self._folds):
                    console.print(
                        f"[bold cyan]Processing fold {i+1}/{self.k}[/bold cyan] (test size: {len(fold_indices)})"
                    )

                    # Process this fold
                    fold_result = self._process_fold(fold_indices, i + 1)
                    results.append(fold_result)

                    # Update progress
                    progress.update("cv_folds", 1)
                    console.print(f"[green]✓[/green] Fold {i+1} complete")

            else:
                console.print(
                    f"[blue]Using parallel processing (n={self.n}, k={self.k}, n_jobs={self.n_jobs})[/blue]"
                )

                parallel_results: List[np.ndarray] = Parallel(
                    n_jobs=self.n_jobs, verbose=0
                )(
                    delayed(self._process_fold)(fold_indices, i + 1)
                    for i, fold_indices in enumerate(self._folds)
                )
                # Update progress after all folds complete
                progress.update("cv_folds", self.k)
                results = parallel_results

            # Concatenate all EIF components
            eif_all = cast(np.ndarray, np.vstack(results))  # Shape: (n, K)

            # Compute point estimates
            v_hat = np.mean(eif_all, axis=0)  # Shape: (K,)

            # Compute covariance matrix (consistent with IPS/SNIPS)
            if self.K == 1:
                # For K=1, ensure we get a 2D array
                Sigma_hat = np.array([[np.var(eif_all[:, 0], ddof=1) / self.n]])
            else:
                Sigma_hat = np.cov(eif_all, rowvar=False) / self.n  # Shape: (K, K)

            # Compute standard errors
            se = np.sqrt(np.diag(Sigma_hat))  # Shape: (K,)

            # Create structured metadata for reliability assessment
            from .reliability import EstimatorMetadata

            structured_metadata = EstimatorMetadata(
                estimator_type="DRCPO",
                k_folds=self.k,
                stabilize_weights=self.stabilize_weights,
                bootstrap_available=eif_all is not None,
            )

            # Add calibration flags to metadata
            structured_metadata.calibrate_weights = self.calibrate_weights
            structured_metadata.calibrate_outcome = self.calibrate_outcome

            # Add weight statistics if available
            if self._weight_stats:
                structured_metadata.ess_values = self._weight_stats["ess_values"]
                structured_metadata.ess_percentage = self._weight_stats[
                    "ess_percentage"
                ]
                structured_metadata.n_clipped = self._weight_stats["n_clipped"]
                structured_metadata.clip_fraction = self._weight_stats["clip_fraction"]
                structured_metadata.weight_range = self._weight_stats["weight_range"]
                structured_metadata.stabilization_applied = self._weight_stats[
                    "stabilization_applied"
                ]

            # Create result object
            result = EstimationResult(
                v_hat=v_hat,
                se=se,
                n=self.n,
                eif_components=eif_all,
                covariance_matrix=Sigma_hat,
                estimator_type="DRCPO",
                n_policies=self.K,
                metadata=structured_metadata.to_dict(),
            )

        console.print(f"[bold green]✓ Cross-validation complete![/bold green]")

        # Write logged target samples to disk if enabled
        self._write_target_samples_log()

        return result
