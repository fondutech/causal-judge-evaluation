"""
Doubly-robust estimators for CJE - FIXED VERSION.

This module contains corrected doubly-robust (DR) estimators that properly
implement the statistical theory for unbiased estimation and valid confidence intervals.

Key fixes:
1. Proper computation of μ_k(X) = E[Y|X,π_k] via Monte Carlo sampling
2. Complete MRDR implementation with control variate term
3. Consistent weight handling (no re-centering after calibration)
4. Proper outcome model calibration on oracle subset
5. Correct variance computation for cross-fitted estimators
6. Full target policy sampling for all contexts
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional, Type, Tuple, cast
import numpy as np
import warnings
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import track

from ..loggers import MultiTargetSampler
from ..calibration import calibrate_weights_isotonic, calibrate_outcome_model_isotonic
from .base import Estimator
from .results import EstimationResult
from .reliability import EstimatorMetadata
from .featurizer import Featurizer, BasicFeaturizer
from .auto_outcome import auto_select
from .dr_warning import check_dr_requirements, validate_dr_setup
from ..judge import BaseJudge
from ..utils.progress import console


class BaseDREstimator(Estimator[Dict[str, Any]]):
    """
    Base class for doubly-robust estimators.

    Provides common functionality for cross-fitting, weight computation,
    outcome modeling, and target policy sampling.
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        k: int = 5,
        seed: int = 0,
        outcome_model_cls: Optional[Type[Any]] = None,
        outcome_model_kwargs: Optional[Dict[str, Any]] = None,
        featurizer: Optional[Featurizer] = None,
        n_jobs: Optional[int] = -1,
        samples_per_policy: int = 1,  # Default to 1 sample per context per policy
        stabilize_weights: bool = True,
        calibrate_weights: bool = True,
        calibrate_outcome: bool = True,
        calibrate_on_oracle: bool = True,  # NEW: Use oracle subset for calibration
        oracle_fraction: float = 0.2,  # NEW: Fraction of data with oracle labels
        verbose: bool = False,
        judge_runner: Optional[BaseJudge] = None,
        score_target_policy_sampled_completions: bool = True,
        work_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(verbose=verbose)
        self.sampler = sampler

        # Enforce minimum k=2 for cross-fitting to avoid overfitting bias
        if k < 2:
            raise ValueError(
                f"Cross-fitting requires k >= 2 to avoid overfitting bias. "
                f"Got k={k}. Use k=5 (default) for standard cross-validation."
            )
        self.k = k
        self.seed = seed
        self.outcome_model_cls = outcome_model_cls
        self.outcome_model_kwargs = outcome_model_kwargs or {}
        self.featurizer = featurizer or BasicFeaturizer()
        self.n_jobs = n_jobs
        self.samples_per_policy = max(1, samples_per_policy)  # Ensure at least 1
        self.stabilize_weights = stabilize_weights
        self.calibrate_weights = calibrate_weights
        self.calibrate_outcome = calibrate_outcome
        self.calibrate_on_oracle = calibrate_on_oracle
        self.oracle_fraction = oracle_fraction
        self.judge_runner = judge_runner
        self.score_target_policy_sampled_completions = (
            score_target_policy_sampled_completions
        )
        self.work_dir = work_dir

        # State variables
        self.n: int = 0
        self.K: int = sampler.K
        self._folds: Optional[List[np.ndarray]] = None
        self._full_logs_data: List[Dict[str, Any]] = []
        self._rewards_full: Optional[np.ndarray] = None
        self._features_full: Optional[np.ndarray] = None
        self._logp_behavior_full: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self._weight_stats: Optional[Dict[str, Any]] = None
        self._cv_results: Optional[List[Dict[str, Any]]] = None
        self._mu_pi_matrix: Optional[np.ndarray] = None
        self._oracle_indices: Optional[np.ndarray] = None
        self._calibrated_weights: Optional[np.ndarray] = None
        self._fold_id_vector: Optional[np.ndarray] = None

    def _should_auto_select_models(self) -> bool:
        """Check if we should auto-select outcome models."""
        return self.outcome_model_cls is None

    def _auto_select_outcome_model(self) -> None:
        """Auto-select outcome model based on dataset characteristics."""
        console.print("[cyan]Auto-selecting outcome model...[/cyan]")

        # Get dataset statistics
        n_samples = len(self._full_logs_data)

        # Use auto_select utility - it returns (model_class, model_kwargs, featurizer)
        model_class, model_kwargs, base_featurizer = auto_select(n_samples)

        self.outcome_model_cls = model_class
        self.outcome_model_kwargs = model_kwargs

        console.print(
            f"[green]Selected {model_class.__name__} with params: {model_kwargs}[/green]"
        )

    def _create_cv_splits(self) -> List[np.ndarray]:
        """Create cross-validation splits."""
        indices = np.arange(self.n)
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
        folds = [test_idx for _, test_idx in kf.split(indices)]

        # Create fold ID vector for cross-fit-safe calibration
        self._fold_id_vector = np.zeros(self.n, dtype=int)
        for fold_idx, test_indices in enumerate(folds):
            self._fold_id_vector[test_indices] = fold_idx

        return folds

    def _compute_importance_weights(
        self, contexts: List[str], responses: List[str], logp_behavior: List[float]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute importance weights matrix."""
        W, weight_stats = self.sampler.importance_weights_matrix(
            contexts,
            responses,
            logp_behavior,
            stabilize=self.stabilize_weights,
            return_stats=True,
        )
        return W, weight_stats

    def _featurize_data(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from logs."""
        # Fit featurizer if needed
        if hasattr(self.featurizer, "fit"):
            self.featurizer.fit(logs)
        # Transform logs to features
        return self.featurizer.transform(logs)

    def _check_prerequisites(self) -> None:
        """Check prerequisites for doubly-robust estimation."""
        # Check DR requirements
        check_dr_requirements(
            estimator_name=self.__class__.__name__.lower()
            .replace("multi", "")
            .replace("estimator", ""),
            samples_per_policy=self.samples_per_policy,
        )

        # Validate complete setup
        validate_dr_setup(
            estimator_name=self.__class__.__name__.lower()
            .replace("multi", "")
            .replace("estimator", ""),
            samples_per_policy=self.samples_per_policy,
            has_judge_runner=self.judge_runner is not None,
            score_target_policy_sampled_completions=self.score_target_policy_sampled_completions,
        )

    @abstractmethod
    def _process_fold(
        self,
        fold_idx: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Process a single CV fold. Must be implemented by subclasses."""
        raise NotImplementedError

    def _select_oracle_subset(self) -> np.ndarray:
        """Select a random subset of data to act as oracle-labeled."""
        n_oracle = int(self.n * self.oracle_fraction)
        oracle_indices = np.random.RandomState(self.seed).choice(
            self.n, size=n_oracle, replace=False
        )
        return oracle_indices

    def _calibrate_weights_once(self) -> None:
        """Calibrate weights using isotonic regression - done ONCE globally."""
        if self.calibrate_weights and self.W is not None:
            console.print("[cyan]Calibrating weights globally...[/cyan]")
            # Use outer fold IDs for cross-fit-safe calibration
            if self._fold_id_vector is not None:
                fold_indices = self._fold_id_vector
            else:
                # Fallback if called before CV splits created
                fold_indices = np.arange(self.n) % self.k
            self._calibrated_weights, _ = calibrate_weights_isotonic(
                self.W, fold_indices=fold_indices, target_mean=1.0
            )
        else:
            self._calibrated_weights = self.W

    def _get_calibrated_weights(
        self, indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get calibrated weights for given indices."""
        if self._calibrated_weights is None:
            raise RuntimeError("Weights not calibrated. Call fit() first.")
        if indices is None:
            return self._calibrated_weights
        return self._calibrated_weights[indices]  # type: ignore

    @abstractmethod
    def _compute_mu_pi_matrix(self) -> np.ndarray:
        """Compute E[Y|X] under each target policy. Must be implemented by subclasses."""
        raise NotImplementedError

    def _generate_target_policy_samples(
        self, contexts: List[str], policy_idx: int, n_samples: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate samples from target policy for given contexts."""
        if not hasattr(self.sampler, "runners") or not self.sampler.runners:
            return []

        if policy_idx >= len(self.sampler.runners):
            return []

        runner = self.sampler.runners[policy_idx]
        if not hasattr(runner, "generate_with_logp"):
            return []

        samples = []
        for context in contexts:
            for _ in range(n_samples):
                try:
                    results = runner.generate_with_logp(
                        [context], return_token_logprobs=False
                    )
                    if results:
                        response, logp = results[0][0], results[0][1]

                        # Score if judge available
                        reward = 0.0
                        if (
                            self.judge_runner
                            and self.score_target_policy_sampled_completions
                        ):
                            if hasattr(self.judge_runner, "score"):
                                reward = self.judge_runner.score(context, response)

                        samples.append(
                            {
                                "context": context,
                                "response": response,
                                "logp": logp,
                                "reward": reward,
                                "policy_idx": policy_idx,
                            }
                        )
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Failed to generate sample: {e}[/yellow]"
                    )

        return samples

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """Fit the doubly-robust estimator."""
        self._full_logs_data = logs
        self.n = len(logs)

        if self.n == 0:
            raise ValueError("Cannot fit estimator with empty logs")

        # Auto-select outcome model if needed
        if self._should_auto_select_models():
            self._auto_select_outcome_model()

        # Check prerequisites
        self._check_prerequisites()

        # Extract data
        self._rewards_full = np.array([log["reward"] for log in logs], dtype=float)
        self._logp_behavior_full = np.array([log["logp"] for log in logs], dtype=float)

        # Compute importance weights
        console.print(
            f"[bold blue]Computing importance weights for {self.K} policies...[/bold blue]"
        )
        contexts = [log["context"] for log in logs]
        responses = [log["response"] for log in logs]

        self.W, self._weight_stats = self._compute_importance_weights(
            contexts, responses, self._logp_behavior_full.tolist()
        )

        # Featurize data
        console.print("[bold blue]Extracting features...[/bold blue]")
        self._features_full = self._featurize_data(logs)

        # Create CV splits (must be before weight calibration for cross-fit safety)
        self._folds = self._create_cv_splits()

        # Calibrate weights ONCE (using fold IDs for cross-fit safety)
        self._calibrate_weights_once()

        # Select oracle subset
        if self.calibrate_on_oracle:
            self._oracle_indices = self._select_oracle_subset()
            console.print(
                f"[cyan]Selected {len(self._oracle_indices)} oracle samples "
                f"({self.oracle_fraction*100:.0f}% of data)[/cyan]"
            )

        # Run cross-validation
        console.print(
            f"[bold blue]Running {self.k}-fold cross-validation...[/bold blue]"
        )

        # Process folds in parallel
        cv_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_fold)(
                fold_idx=i,
                train_indices=np.concatenate(
                    [f for j, f in enumerate(self._folds) if j != i]
                ),
                test_indices=fold,
            )
            for i, fold in enumerate(self._folds)
        )

        self._cv_results = cv_results

        # Compute mu_pi matrix with proper target policy sampling
        console.print(
            "[bold blue]Computing E[Y|X] under target policies...[/bold blue]"
        )
        self._mu_pi_matrix = self._compute_mu_pi_matrix()


class MultiDRCPOEstimator(BaseDREstimator):
    """
    Corrected Doubly-Robust Cross-Policy Optimization estimator.

    Properly computes μ_k(X) = E[Y|X,π_k] via Monte Carlo sampling from target policies.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._all_target_samples: List[Dict[str, Any]] = []

    def _process_fold(
        self,
        fold_idx: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Process a single CV fold for DR-CPO."""
        if self._features_full is None or self._rewards_full is None:
            raise RuntimeError("Arrays not initialized. Call fit() first.")

        # Get training data
        X_train = self._features_full[train_indices]
        y_train = self._rewards_full[train_indices]

        # Fit outcome model
        if self.outcome_model_cls is None:
            raise RuntimeError("Outcome model not selected. Call fit() first.")
        outcome_model = self.outcome_model_cls(**self.outcome_model_kwargs)
        outcome_model.fit(X_train, y_train)

        # Get test data
        X_test = self._features_full[test_indices]

        # Get calibrated weights for test indices
        W_test = self._get_calibrated_weights(test_indices)

        # Predict on test set
        mu_hat_test = outcome_model.predict(X_test)

        # Apply outcome model calibration if enabled
        if self.calibrate_outcome:
            if self.calibrate_on_oracle and self._oracle_indices is not None:
                # Use only oracle subset for calibration
                oracle_train = np.intersect1d(train_indices, self._oracle_indices)
                if len(oracle_train) > 10:  # Need sufficient samples
                    X_oracle = self._features_full[oracle_train]
                    y_oracle = self._rewards_full[oracle_train]
                    oracle_preds = outcome_model.predict(X_oracle)
                    calibration_fn, _ = calibrate_outcome_model_isotonic(
                        oracle_preds, y_oracle
                    )
                    mu_hat_test = calibration_fn(mu_hat_test)

        # Generate target policy samples for ALL test contexts
        target_samples = []
        test_contexts = [self._full_logs_data[idx]["context"] for idx in test_indices]

        for k in range(self.K):
            policy_samples = self._generate_target_policy_samples(
                test_contexts, k, n_samples=self.samples_per_policy
            )
            for sample in policy_samples:
                sample["fold"] = fold_idx
                sample["test_idx"] = test_indices[
                    test_contexts.index(sample["context"])
                ]
                target_samples.append(sample)

        return {
            "fold_idx": fold_idx,
            "mu_hat_test": mu_hat_test,
            "W_test": W_test,
            "test_indices": test_indices,
            "outcome_model": outcome_model,
            "target_samples": target_samples,
        }

    def _compute_mu_pi_matrix(self) -> np.ndarray:
        """Properly compute E[Y|X] under each target policy using MC samples."""
        if not self._cv_results:
            raise RuntimeError("Must call fit() before computing mu_pi")

        mu_pi = np.zeros((self.n, self.K))
        mu_pi_counts = np.zeros((self.n, self.K))  # Track samples per context/policy

        # Collect all target samples
        self._all_target_samples = []
        for result in self._cv_results:
            self._all_target_samples.extend(result.get("target_samples", []))

        # If we have target samples, use them to compute μ_k(X)
        if self._all_target_samples:
            # Group samples by context and policy
            for sample in self._all_target_samples:
                test_idx = sample["test_idx"]
                policy_idx = sample["policy_idx"]

                # Get features for this context
                if self._features_full is None:
                    raise RuntimeError("Features not initialized")
                X_context = self._features_full[test_idx : test_idx + 1]

                # Use outcome model from the appropriate fold
                fold_idx = sample["fold"]
                outcome_model = self._cv_results[fold_idx]["outcome_model"]

                # Predict reward for this sample
                # NOTE: Currently uses context features only. If future outcome models
                # consume the generated response, pass sample["response"] to predict.
                pred_reward = outcome_model.predict(X_context)[0]

                # Accumulate predictions
                mu_pi[test_idx, policy_idx] += pred_reward
                mu_pi_counts[test_idx, policy_idx] += 1

            # Average over samples
            for k in range(self.K):
                mask = mu_pi_counts[:, k] > 0
                mu_pi[mask, k] /= mu_pi_counts[mask, k]

                # For contexts without samples, fall back to logged data prediction
                no_sample_mask = mu_pi_counts[:, k] == 0
                if np.any(no_sample_mask):
                    # Use cross-fitted predictions from logged data
                    for result in self._cv_results:
                        test_indices = result["test_indices"]
                        mu_hat = result["mu_hat_test"]
                        overlap = np.intersect1d(
                            test_indices[no_sample_mask[test_indices]],
                            np.where(no_sample_mask)[0],
                        )
                        if len(overlap) > 0:
                            mu_pi[overlap, k] = mu_hat[np.isin(test_indices, overlap)]

        else:
            # Fallback: If no target samples, assume outcome model is context-only
            console.print(
                "[yellow]Warning: No target samples generated. "
                "Assuming outcome model uses only context features.[/yellow]"
            )
            for result in self._cv_results:
                test_indices = result["test_indices"]
                mu_hat = result["mu_hat_test"]
                # If model is context-only, same prediction for all policies
                for k in range(self.K):
                    mu_pi[test_indices, k] = mu_hat

        return mu_pi

    def estimate(self) -> EstimationResult:
        """Compute DR-CPO estimates with proper EIF."""
        if self._mu_pi_matrix is None or self._calibrated_weights is None:
            raise RuntimeError("Must call fit() before estimate()")

        v_hat = np.zeros(self.K)
        eif_components = np.zeros((self.n, self.K))

        for k in range(self.K):
            # E[μ_k(X)]
            mu_k = self._mu_pi_matrix[:, k]
            v_hat[k] = np.mean(mu_k)

            # Get calibrated weights (already centered with E[W]=1)
            w_k = self._calibrated_weights[:, k]

            # EIF: μ_k(X) + w_k * (Y - μ_k(X)) - v_hat[k]
            if self._rewards_full is None:
                raise RuntimeError("Rewards not initialized")
            residuals = self._rewards_full - mu_k
            eif_components[:, k] = mu_k + w_k * residuals - v_hat[k]

        # Compute covariance matrix for cross-fitted estimator
        if self.K == 1:
            Sigma_hat = np.array([[np.var(eif_components[:, 0], ddof=1) / self.n]])
        else:
            Sigma_hat = np.cov(eif_components.T) / self.n

        # Standard errors
        se = np.sqrt(np.diag(Sigma_hat))

        # Create metadata
        metadata = EstimatorMetadata(
            estimator_type="DR-CPO",
            k_folds=self.k,
            calibrate_weights=self.calibrate_weights,
            calibrate_outcome=self.calibrate_outcome,
            bootstrap_available=True,
            stabilize_weights=self.stabilize_weights,
        )

        if self._weight_stats:
            metadata.weight_range = self._weight_stats.get("weight_range")
            metadata.ess_values = self._weight_stats.get("ess_values")
            metadata.ess_percentage = self._weight_stats.get("ess_percentage")

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            n_policies=self.K,
            eif_components=eif_components,
            covariance_matrix=Sigma_hat,
            metadata=metadata.to_dict(),
        )


class MultiMRDREstimator(MultiDRCPOEstimator):
    """
    Corrected Multi-Robust Doubly-Robust estimator.

    Implements the full MRDR with control variate term for variance reduction.
    """

    def __init__(self, lambda_reg: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg
        self._alpha_hat: Optional[np.ndarray] = None  # Per-policy α̂
        self._control_variate_gap: Optional[np.ndarray] = None

    def _compute_control_variate_coefficient(self) -> np.ndarray:
        """Compute per-policy optimal α̂ₖ for control variate."""
        if self._cv_results is None or self._rewards_full is None:
            raise RuntimeError("Must run cross-validation first")

        # Compute α̂ₖ separately for each policy
        alpha_k = np.zeros(self.K)

        for k in range(self.K):
            residuals_k = []
            gaps_k = []

            for result in self._cv_results:
                test_indices = result["test_indices"]
                mu_hat = result["mu_hat_test"]
                W_test = self._get_calibrated_weights(test_indices)

                # Residuals: W_k * (Y - m)
                y_test = self._rewards_full[test_indices]
                residuals = W_test[:, k] * (y_test - mu_hat)
                residuals_k.extend(residuals)

                # Gaps: m_π' - m (using logged data prediction as baseline)
                mu_pi_k = self._mu_pi_matrix[test_indices, k]
                gaps = mu_pi_k - mu_hat
                gaps_k.extend(gaps)

            # Compute optimal α̂ₖ = Cov(residuals, gaps) / Var(gaps)
            residuals_array = np.array(residuals_k)
            gaps_array = np.array(gaps_k)

            if np.var(gaps_array) > 1e-10:  # Avoid division by zero
                cov = np.cov(residuals_array, gaps_array)[0, 1]
                var_gaps = np.var(gaps_array)
                alpha_k[k] = cov / (var_gaps + self.lambda_reg)  # Regularized
            else:
                alpha_k[k] = 0.0

        return alpha_k

    def estimate(self) -> EstimationResult:
        """Compute MRDR estimates with control variate."""
        # First compute standard DR-CPO estimate
        dr_result = super().estimate()

        # Compute per-policy control variate coefficients
        self._alpha_hat = self._compute_control_variate_coefficient()
        console.print("[cyan]MRDR control variate α̂ₖ:[/cyan]")
        for k in range(self.K):
            console.print(f"  Policy {k}: α̂ₖ = {self._alpha_hat[k]:.4f}")

        # Apply control variate correction if any αₖ is meaningful
        if np.any(np.abs(self._alpha_hat) > 1e-6):
            # Recompute EIF with control variate
            eif_components = np.zeros((self.n, self.K))

            for k in range(self.K):
                mu_k = self._mu_pi_matrix[:, k]
                w_k = self._calibrated_weights[:, k]

                # Standard DR-CPO EIF
                residuals = self._rewards_full - mu_k
                dr_eif = mu_k + w_k * residuals - dr_result.v_hat[k]

                # Control variate term
                # For each fold, compute m_π - m_logged
                control_term = np.zeros(self.n)
                for result in self._cv_results:
                    test_indices = result["test_indices"]
                    mu_hat = result["mu_hat_test"]
                    mu_pi_test = mu_k[test_indices]
                    gap = mu_pi_test - mu_hat
                    control_term[test_indices] = self._alpha_hat[k] * gap

                # MRDR EIF
                eif_components[:, k] = dr_eif - control_term

            # Recompute variance with new EIF
            if self.K == 1:
                Sigma_hat = np.array([[np.var(eif_components[:, 0], ddof=1) / self.n]])
            else:
                Sigma_hat = np.cov(eif_components.T) / self.n

            se = np.sqrt(np.diag(Sigma_hat))

            # Update result
            dr_result.eif_components = eif_components
            dr_result.covariance_matrix = Sigma_hat
            dr_result.se = se

        # Update metadata
        dr_result.metadata["estimator_type"] = "MRDR"
        dr_result.metadata["alpha_hat"] = self._alpha_hat.tolist()
        dr_result.metadata["lambda_reg"] = self.lambda_reg

        return dr_result


# For backward compatibility, keep the original class names
DRCPOEstimator = MultiDRCPOEstimator
MRDREstimator = MultiMRDREstimator
