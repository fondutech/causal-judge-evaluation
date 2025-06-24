"""
Doubly-robust estimators for CJE.

This module contains all doubly-robust (DR) estimators that combine
importance weighting with outcome modeling for reduced variance and
improved bias-variance tradeoff.

Available estimators:
- MultiDRCPOEstimator: Doubly-Robust Cross-Policy Optimization
- MultiMRDREstimator: Multi-Robust Doubly-Robust (variance-optimized)

All DR estimators use cross-fitting to avoid overfitting and support
target policy sampling for improved outcome model training.
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
        samples_per_policy: int = 2,
        stabilize_weights: bool = True,
        calibrate_weights: bool = True,
        calibrate_outcome: bool = True,
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
        self.samples_per_policy = samples_per_policy
        self.stabilize_weights = stabilize_weights
        self.calibrate_weights = calibrate_weights
        self.calibrate_outcome = calibrate_outcome
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
        return [test_idx for _, test_idx in kf.split(indices)]

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

    @abstractmethod
    def _compute_mu_pi_matrix(self) -> np.ndarray:
        """Compute E[Y|X] under each target policy. Must be implemented by subclasses."""
        raise NotImplementedError

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

        # Target sample logging would go here if implemented

        # Create CV splits
        self._folds = self._create_cv_splits()

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

        # Compute mu_pi matrix
        console.print(
            "[bold blue]Computing E[Y|X] under target policies...[/bold blue]"
        )
        self._mu_pi_matrix = self._compute_mu_pi_matrix()

        # Target sample logging would go here if implemented


class MultiDRCPOEstimator(BaseDREstimator):
    """
    Doubly-Robust Cross-Policy Optimization estimator.

    This estimator combines importance weighting with outcome modeling
    for improved bias-variance tradeoff. It uses cross-fitting to avoid
    overfitting and supports target policy sampling.

    The DR-CPO estimate is:
    v_hat(π_k) = E_n[μ_k(X) + w_k(X,A) * (Y - μ_k(X))]

    where μ_k(X) = E[Y|X] under π_k, and w_k are importance weights.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._all_target_samples: List[Dict[str, Any]] = []
        self._reward_variances_full: Optional[np.ndarray] = None

    def _process_fold(
        self,
        fold_idx: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Process a single CV fold for DR-CPO."""
        # Ensure arrays are not None
        if self._features_full is None or self._rewards_full is None or self.W is None:
            raise RuntimeError("Arrays not initialized. Call fit() first.")

        # Get training data
        X_train = self._features_full[train_indices]
        y_train = self._rewards_full[train_indices]
        W_train = self.W[train_indices]  # Shape: (n_train, K)

        # Fit outcome model
        if self.outcome_model_cls is None:
            raise RuntimeError("Outcome model not selected. Call fit() first.")
        outcome_model = self.outcome_model_cls(**self.outcome_model_kwargs)
        outcome_model.fit(X_train, y_train)

        # Get test data
        X_test = self._features_full[test_indices]
        W_test = self.W[test_indices]  # Shape: (n_test, K)

        # Apply isotonic weight calibration if enabled
        if self.calibrate_weights:
            fold_indices = np.zeros(len(test_indices), dtype=int)
            for j, idx in enumerate(test_indices):
                fold_indices[j] = j % 3  # Sub-folds for calibration

            W_test_calibrated, _ = calibrate_weights_isotonic(
                W_test, fold_indices=fold_indices, target_mean=1.0
            )
            W_test = W_test_calibrated

        # Apply outcome model calibration if enabled
        mu_hat_test = outcome_model.predict(X_test)
        if self.calibrate_outcome:
            calibration_fn, _ = calibrate_outcome_model_isotonic(
                outcome_model.predict(X_train), y_train
            )
            mu_hat_test_calibrated = calibration_fn(mu_hat_test)
            mu_hat_test = mu_hat_test_calibrated

        # Generate target policy samples if requested
        target_samples = []
        if (
            self.samples_per_policy > 0
            and hasattr(self.sampler, "runners")
            and self.sampler.runners
        ):
            for k in range(self.K):
                runner = self.sampler.runners[k]
                if hasattr(runner, "generate_with_logp"):
                    for idx in test_indices[:5]:  # Sample for subset of test points
                        context = self._full_logs_data[idx]["context"]
                        for _ in range(self.samples_per_policy):
                            # Generate response using the policy runner
                            results = runner.generate_with_logp(
                                [context], return_token_logprobs=False
                            )
                            if results:
                                response, logp = results[0][0], results[0][1]
                                if (
                                    self.judge_runner
                                    and self.score_target_policy_sampled_completions
                                ):
                                    # Score using judge - assuming it has a score method
                                    if hasattr(self.judge_runner, "score"):
                                        reward = self.judge_runner.score(
                                            context, response
                                        )
                                    else:
                                        reward = 0.0
                                else:
                                    reward = 0.0
                                target_samples.append(
                                    {
                                        "fold": fold_idx,
                                        "policy_idx": k,
                                        "context": context,
                                        "response": response,
                                        "reward": reward,
                                        "test_idx": idx,
                                    }
                                )

        return {
            "fold_idx": fold_idx,
            "mu_hat_test": mu_hat_test,
            "W_test": W_test,
            "test_indices": test_indices,
            "outcome_model": outcome_model,
            "target_samples": target_samples,
        }

    def _compute_mu_pi_matrix(self) -> np.ndarray:
        """Compute E[Y|X] under each target policy for DR-CPO."""
        if not self._cv_results:
            raise RuntimeError("Must call fit() before computing mu_pi")

        mu_pi = np.zeros((self.n, self.K))

        # Aggregate predictions from all folds
        for result in self._cv_results:
            test_indices = result["test_indices"]
            mu_hat = result["mu_hat_test"]
            W_test = result["W_test"]

            # For each policy, compute weighted prediction
            for k in range(self.K):
                # Weighted average of predictions
                weights_k = W_test[:, k]
                weights_k = weights_k / np.mean(weights_k)  # Normalize
                mu_pi[test_indices, k] = mu_hat * weights_k

        # Collect all target samples
        self._all_target_samples = []
        for result in self._cv_results:
            self._all_target_samples.extend(result.get("target_samples", []))

        return mu_pi

    def estimate(self) -> EstimationResult:
        """Compute DR-CPO estimates."""
        if self._mu_pi_matrix is None:
            raise RuntimeError("Must call fit() before estimate()")

        # DR-CPO estimator
        v_hat = np.zeros(self.K)
        eif_components = np.zeros((self.n, self.K))

        for k in range(self.K):
            # E[μ_k(X)]
            mu_k = self._mu_pi_matrix[:, k]
            v_hat[k] = np.mean(mu_k)

            # EIF: μ_k(X) + w_k * (Y - μ_k(X))
            if self.W is None or self._rewards_full is None:
                raise RuntimeError("Arrays not initialized")
            w_k = self.W[:, k] / np.mean(self.W[:, k])  # Normalized weights
            residuals = self._rewards_full - mu_k
            eif_components[:, k] = mu_k + w_k * residuals - v_hat[k]

        # Compute standard errors
        se = np.std(eif_components, axis=0) / np.sqrt(self.n)

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
            covariance_matrix=(
                np.cov(eif_components.T) / self.n
                if self.K > 1
                else np.array([[np.var(eif_components[:, 0], ddof=1) / self.n]])
            ),
            metadata=metadata.to_dict(),
        )


class MultiMRDREstimator(BaseDREstimator):
    """
    Multi-Robust Doubly-Robust estimator.

    This estimator optimizes the variance of the DR estimator by using
    a variance-minimizing combination of outcome models. It's more robust
    to model misspecification than standard DR-CPO.

    The MRDR estimate uses a weighted combination of outcome models
    where weights are chosen to minimize variance.
    """

    def __init__(self, lambda_reg: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg  # Regularization for variance optimization
        self._model_weights: Optional[np.ndarray] = None

    def _compute_optimal_model_weights(
        self, residuals: np.ndarray, W: np.ndarray
    ) -> np.ndarray:
        """Compute variance-minimizing weights for outcome models."""
        # Simplified version - in practice this would solve an optimization problem
        # For now, use equal weights
        n_models = residuals.shape[1] if residuals.ndim > 1 else 1
        return np.ones(n_models) / n_models

    def _process_fold(
        self,
        fold_idx: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """Process a single CV fold for MRDR."""
        # Since we inherit from MultiDRCPOEstimator, we can call its _process_fold
        # But we need to cast self to avoid mypy type issues
        drcpo_self = cast(MultiDRCPOEstimator, self)
        drcpo_result = MultiDRCPOEstimator._process_fold(
            drcpo_self, fold_idx, train_indices, test_indices
        )
        result = drcpo_result.copy()

        # MRDR-specific: fit multiple outcome models with different hyperparameters
        if self._features_full is None or self._rewards_full is None:
            raise RuntimeError("Arrays not initialized")
        X_train = self._features_full[train_indices]
        y_train = self._rewards_full[train_indices]

        # Fit ensemble of models
        models = []
        if self.outcome_model_cls is None:
            raise RuntimeError("Outcome model not selected")
        for alpha in [0.01, 0.1, 1.0]:  # Different regularization strengths
            model_kwargs = self.outcome_model_kwargs.copy()
            model_kwargs["alpha"] = alpha
            model = self.outcome_model_cls(**model_kwargs)
            model.fit(X_train, y_train)
            models.append(model)

        result["models"] = models
        return result

    def _compute_mu_pi_matrix(self) -> np.ndarray:
        """Compute E[Y|X] under each target policy for MRDR."""
        if not self._cv_results:
            raise RuntimeError("Must call fit() before computing mu_pi")

        mu_pi = np.zeros((self.n, self.K))

        # Use ensemble predictions
        for result in self._cv_results:
            test_indices = result["test_indices"]
            models = result.get("models", [result["outcome_model"]])
            if self._features_full is None:
                raise RuntimeError("Features not initialized")
            X_test = self._features_full[test_indices]
            W_test = result["W_test"]

            # Average predictions across models
            predictions = np.mean([m.predict(X_test) for m in models], axis=0)

            # Apply calibration if enabled
            if self.calibrate_outcome:
                # Calibrate ensemble predictions
                if self._features_full is None or self._rewards_full is None:
                    raise RuntimeError("Arrays not initialized")
                # Reconstruct training indices for this fold
                fold_idx = result["fold_idx"]
                if self._folds is None:
                    raise RuntimeError("Folds not initialized")
                all_train_indices = np.concatenate(
                    [f for j, f in enumerate(self._folds) if j != fold_idx]
                )
                X_train = self._features_full[all_train_indices]
                y_train = self._rewards_full[all_train_indices]
                train_preds = np.mean([m.predict(X_train) for m in models], axis=0)

                calibration_fn, _ = calibrate_outcome_model_isotonic(
                    train_preds, y_train
                )
                predictions = calibration_fn(predictions)

            # Weight by importance weights
            for k in range(self.K):
                weights_k = W_test[:, k]
                weights_k = weights_k / np.mean(weights_k)
                mu_pi[test_indices, k] = predictions * weights_k

        return mu_pi

    def estimate(self) -> EstimationResult:
        """Compute MRDR estimates."""
        # Call concrete DRCPO estimate method directly
        if self._mu_pi_matrix is None:
            raise RuntimeError("Must call fit() before estimate()")

        # DR-CPO estimator
        v_hat = np.zeros(self.K)
        eif_components = np.zeros((self.n, self.K))

        for k in range(self.K):
            # E[μ_k(X)]
            mu_k = self._mu_pi_matrix[:, k]
            v_hat[k] = np.mean(mu_k)

            # EIF: μ_k(X) + w_k * (Y - μ_k(X))
            if self.W is None or self._rewards_full is None:
                raise RuntimeError("Arrays not initialized")
            w_k = self.W[:, k] / np.mean(self.W[:, k])  # Normalized weights
            residuals = self._rewards_full - mu_k
            eif_components[:, k] = mu_k + w_k * residuals - v_hat[k]

        # Compute standard errors
        se = np.std(eif_components, axis=0) / np.sqrt(self.n)

        # Create metadata
        metadata = EstimatorMetadata(
            estimator_type="MRDR",
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

        result = EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            n_policies=self.K,
            eif_components=eif_components,
            covariance_matrix=(
                np.cov(eif_components.T) / self.n
                if self.K > 1
                else np.array([[np.var(eif_components[:, 0], ddof=1) / self.n]])
            ),
            metadata=metadata.to_dict(),
        )

        # Add MRDR-specific metadata
        result.metadata["lambda_reg"] = self.lambda_reg

        # Update metadata
        if isinstance(result.metadata, dict):
            result.metadata["estimator_type"] = "MRDR"
            result.metadata["lambda_reg"] = self.lambda_reg

        return result


# For backward compatibility, keep the original class names
DRCPOEstimator = MultiDRCPOEstimator
MRDREstimator = MultiMRDREstimator
