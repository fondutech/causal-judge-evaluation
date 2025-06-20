"""
More-Robust Doubly Robust (MRDR) estimators.

This module implements the MRDR estimator from Kallus & Uehara (ICML 2020):
"Double Reinforcement Learning for Off-Policy Evaluation"

MRDR chooses the outcome model m in a data-dependent way that minimizes the variance
of the standard Doubly-Robust estimator while preserving its unbiasedness.

**Architecture:**
- MultiMRDREstimator: Native multi-policy implementation with vectorized outputs
- The CLI automatically uses the unified multi-policy approach
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, Union, cast
import numpy as np
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
from .base import Estimator
from .results import EstimationResult
from ..loggers.multi_target_sampler import MultiTargetSampler
from .featurizer import Featurizer, BasicFeaturizer
from .auto_outcome import auto_select

# Define a type alias for scikit-learn compatible models
ModelType = Any


class MultiMRDREstimator(Estimator[Dict[str, Any]]):
    """
    Vectorised MRDR for K target policies.

    This estimator implements the More-Robust Doubly Robust approach that chooses
    the outcome model to minimize the variance of the DR estimator while preserving
    unbiasedness. It extends MRDR to evaluate multiple target policies
    {π¹, π², ..., πᴷ} simultaneously.

    The key insight is to solve a weighted least squares problem where the weights
    involve importance ratios, leading to better alignment between the regression
    residuals and importance weights.

    Args:
        sampler: MultiTargetSampler instance for K target policies (required)
        k: Number of cross-validation folds (default: 5)
        seed: Random seed for shuffling data (default: 0)
        outcome_model_cls: The class of the outcome model to use (default: Ridge)
        outcome_model_kwargs: Keyword arguments for instantiating the outcome model
        featurizer: Featurizer instance for transforming log items (default: BasicFeaturizer)
        n_jobs: Number of parallel jobs for cross-validation (default: -1, uses all processors)
        samples_per_policy: Number of samples per policy when computing μ_πᵏ(x) (default: 2)
            Setting to 2 provides good variance reduction and estimates absolute target policy values.
            Setting to 0 provides speedup but changes the interpretation: estimates become differences
            from the outcome model baseline rather than absolute target policy values.
            Increase (e.g., to 5-10) if you need maximum variance reduction.

            ⚠️ IMPORTANT: samples_per_policy=0 fundamentally changes what v_hat represents:
            - samples_per_policy > 0: v_hat = absolute expected value of target policy
            - samples_per_policy = 0: v_hat = difference between target policy and outcome model baseline

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
        verbose: Verbose flag for progress prints (default: False)
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        k: int = 5,
        seed: int = 0,
        outcome_model_cls: Type[ModelType] = Ridge,
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
        super().__init__()
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
        self.verbose = verbose
        self.judge_runner = judge_runner
        self.score_target_policy_sampled_completions = (
            score_target_policy_sampled_completions
        )
        self.work_dir = work_dir

        # Storage for target samples (logged automatically)
        self._target_sample_logs: List[Dict[str, Any]] = []

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
        self._weight_stats: Optional[Dict[str, Any]] = (
            None  # Weight statistics for reliability assessment
        )

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """
        Fit k-fold cross-validated multi-policy MRDR estimator.

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

        # Extract basic data
        self._rewards_full = np.array([log["reward"] for log in logs], dtype=float)
        self._logp_behavior_full = np.array([log["logp"] for log in logs], dtype=float)

        # Extract reward variances from unified score format
        from ..utils.score_storage import ScoreCompatibilityLayer

        self._reward_variances_full = np.array(
            [
                (
                    ScoreCompatibilityLayer.get_score_variance(log, "reward")
                    if "reward" in log
                    else log.get("reward_variance", 0.0)
                )
                for log in logs
            ],
            dtype=float,
        )

        # Compute importance weights matrix W (n, K) with statistics
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
        from cje.utils.progress import console

        if self.calibrate_weights:
            console.print(
                "[bold blue]✓ Isotonic weight calibration enabled for MRDR[/bold blue]"
            )
        if self.calibrate_outcome:
            console.print(
                "[bold blue]✓ Isotonic outcome calibration enabled for MRDR[/bold blue]"
            )

        # ------------------------------------------------------------------
        # Automatic outcome-model/featurizer selection (shared helper)
        # ------------------------------------------------------------------
        if (
            self.outcome_model_cls is Ridge  # default class
            and not self.outcome_model_kwargs
            and isinstance(self.featurizer, BasicFeaturizer)
        ):
            (
                self.outcome_model_cls,
                self.outcome_model_kwargs,
                base_featurizer,
            ) = auto_select(self.n)

            # If judge_runner is provided, wrap with ScoreAugmentFeaturizer to include judge scores
            if self.judge_runner is not None:
                from .auto_outcome import ScoreAugmentFeaturizer

                # Include variance information from unified judge system
                self.featurizer = ScoreAugmentFeaturizer(
                    base_featurizer, include_variance=True
                )
            else:
                self.featurizer = base_featurizer

        # Fit featurizer and transform logs
        self.featurizer.fit(logs)
        self._features_full = self.featurizer.transform(logs)

        # Set up cross-validation folds
        indices = np.arange(self.n)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        self._folds = np.array_split(indices, self.k)

    def _fit_mrdr_outcome_model(
        self, X_train: np.ndarray, y_train: np.ndarray, W_train: np.ndarray
    ) -> Any:
        """
        Fit the MRDR outcome model using variance-minimizing weighted least squares.

        The key insight of MRDR is to solve:
        min_m E[(m(X) + w(R - m(X,S)))²]

        This leads to a weighted least squares problem where the sample weights are
        related to the importance ratios.

        Args:
            X_train: Training features (n_train, d)
            y_train: Training rewards (n_train,)
            W_train: Training importance weights (n_train, K)

        Returns:
            Fitted outcome model
        """
        # For multi-policy MRDR, we need to handle K policies
        # We'll fit a single outcome model but weight by the average importance weights
        # This is a reasonable approximation for the multi-policy case

        # Compute average importance weights across policies
        w_avg = np.mean(W_train, axis=1)  # Shape: (n_train,)

        # MRDR weights: The optimal weights for variance minimization
        # In the original paper, these are derived from the variance minimization
        # For practical implementation, we use w_i^2 as the sample weights
        # This aligns the regression with the importance sampling variance structure
        sample_weights = w_avg**2

        # Normalize weights to avoid numerical issues
        sample_weights = sample_weights / np.mean(sample_weights)

        # Fit weighted regression
        model = self.outcome_model_cls(**self.outcome_model_kwargs)

        # Check if the model supports sample weights
        if hasattr(model, "fit") and "sample_weight" in model.fit.__code__.co_varnames:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            # Fallback to unweighted if sample_weight not supported
            model.fit(X_train, y_train)

        return model

    def _compute_mu_pi_matrix(
        self, log_items: List[Dict[str, Any]], outcome_model: Any
    ) -> np.ndarray:
        """
        Compute μ_πᵏ(x) = E_πᵏ[μ(x,s)] for all K policies using live sampling.

        Args:
            log_items: List of log entries
            outcome_model: Fitted outcome model

        Returns:
            Matrix of shape (len(log_items), K) with μ_πᵏ(x) values
        """
        n_items = len(log_items)
        mu_pi_matrix = np.zeros((n_items, self.K))

        for i, log_item in enumerate(log_items):
            context = str(log_item["context"])
            # Use configured number of samples per policy (avoid hard-coded 5)
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
                                from cje.utils.progress import console

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
                                "fold_label": "MRDR",
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

            from cje.utils.progress import console

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
            from cje.utils.progress import console

            console.print(
                f"[yellow]⚠️  Failed to write target samples log: {e}[/yellow]"
            )

    def _process_fold(self, fold_indices: np.ndarray) -> np.ndarray:
        """
        Process a single cross-validation fold.

        This method is designed to be called in parallel by joblib.

        Args:
            fold_indices: Indices for the test fold

        Returns:
            EIF components for this fold (n_test, K)
        """
        # Determine train indices (all folds except current)
        assert self._folds is not None  # Type assertion for mypy
        train_indices = np.concatenate(
            [f_idx for f_idx in self._folds if not np.array_equal(f_idx, fold_indices)]
        )
        test_indices = fold_indices

        # Train MRDR outcome model on training fold
        assert self._features_full is not None  # Type assertion for mypy
        assert self._rewards_full is not None  # Type assertion for mypy
        assert self.W is not None  # Type assertion for mypy
        X_train = self._features_full[train_indices]
        y_train = self._rewards_full[train_indices]
        W_train = self.W[train_indices]

        if X_train.shape[0] > 0:
            mu = self._fit_mrdr_outcome_model(X_train, y_train, W_train)
        else:
            # Fallback for empty training fold
            mu = self.outcome_model_cls(**self.outcome_model_kwargs)
            mu.fit(X_train, y_train)

        # Evaluate on test fold
        assert self._full_logs_data is not None  # Type assertion for mypy
        test_log_items = [self._full_logs_data[idx] for idx in test_indices]

        # Compute μ(x,y) for observed (context, response) pairs
        X_test = self._features_full[test_indices]
        mu_hat_test = mu.predict(X_test)  # Shape: (n_test,)

        # Compute μ_πᵏ(x) for all K policies
        mu_pi_test = self._compute_mu_pi_matrix(
            test_log_items, mu
        )  # Shape: (n_test, K)

        # Get importance weights and rewards for test fold
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
            train_preds = mu.predict(self._features_full[train_indices])
            train_rewards = self._rewards_full[train_indices]

            calibration_fn, outcome_diagnostics = calibrate_outcome_model_isotonic(
                train_preds, train_rewards
            )

            # Apply calibration to test predictions
            mu_hat_test_calibrated = calibration_fn(mu_hat_test)
        else:
            mu_hat_test_calibrated = mu_hat_test

        # Compute EIF components: μ_πᵏ(x) + wᵏ * (r - μ(x,y))
        # With calibration: uses calibrated weights and/or calibrated outcome model
        eif_test = mu_pi_test + W_test_calibrated * (
            r_test[:, np.newaxis] - mu_hat_test_calibrated[:, np.newaxis]
        )

        return cast(np.ndarray, eif_test)

    def estimate(self) -> EstimationResult:
        """
        Return the multi-policy MRDR estimates.

        Returns:
            EstimationResult containing:
                v_hat: Point estimates (K,)
                se: Standard errors (K,)
                cov: Covariance matrix (K, K)
                n: Number of samples (scalar)
                K: Number of policies (scalar)
                name: Estimator name
        """
        if (
            self._folds is None
            or self.W is None
            or self._rewards_full is None
            or self._features_full is None
        ):
            raise RuntimeError("Must call fit() before estimate()")

        # Decide whether to use parallel processing
        use_parallel = (
            self.n_jobs not in (None, 1)
            and self.n >= 100  # large enough to amortise overhead
            and self.k > 2
        )

        if self.verbose:
            from cje.utils.progress import console

            console.print(
                f"[dim]MRDR using {'parallel' if use_parallel else 'sequential'} processing (n_jobs={self.n_jobs})[/dim]"
            )

        if use_parallel:
            all_eif_components = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._process_fold)(fold_indices)
                for fold_indices in self._folds
            )
        else:
            # Sequential fallback to avoid joblib overhead and potential deadlocks
            all_eif_components = [
                self._process_fold(fold_indices) for fold_indices in self._folds
            ]

        # Concatenate all EIF components
        eif_all = cast(np.ndarray, np.vstack(all_eif_components))  # Shape: (n, K)

        # Compute point estimates
        v_hat = np.mean(eif_all, axis=0)  # Shape: (K,)

        # Compute covariance matrix
        if self.K == 1:
            # For K=1, ensure we get a 2D array
            Sigma_hat = np.array([[np.var(eif_all[:, 0], ddof=1) / self.n]])
        else:
            Sigma_hat = np.cov(eif_all, rowvar=False) / self.n  # Shape: (K, K)

        # Compute standard errors
        se = np.sqrt(np.diag(Sigma_hat))  # Shape: (K,)

        # Write logged target samples to disk if enabled
        # Write logged target samples to disk
        self._write_target_samples_log()

        # Create structured metadata for reliability assessment
        from .reliability import EstimatorMetadata

        structured_metadata = EstimatorMetadata(
            estimator_type="MRDR",
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
            structured_metadata.ess_percentage = self._weight_stats["ess_percentage"]
            structured_metadata.n_clipped = self._weight_stats["n_clipped"]
            structured_metadata.clip_fraction = self._weight_stats["clip_fraction"]
            structured_metadata.weight_range = self._weight_stats["weight_range"]
            structured_metadata.stabilization_applied = self._weight_stats[
                "stabilization_applied"
            ]

        # This is the same as DR-CPO, but with the variance-optimized outcome model
        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            eif_components=eif_all,
            covariance_matrix=Sigma_hat,
            estimator_type="MRDR",
            n_policies=self.K,
            metadata=structured_metadata.to_dict(),
        )
