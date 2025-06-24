"""
Base class for cross-fitted estimators to eliminate code duplication.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type, cast
import numpy as np
import warnings
from joblib import Parallel, delayed

from .base import Estimator
from .results import EstimationResult
from ..loggers.multi_target_sampler import MultiTargetSampler
from .featurizer import Featurizer, BasicFeaturizer
from .auto_outcome import auto_select
from .reliability import EstimatorMetadata
from ..utils.progress import ProgressMonitor, console


class BaseCrossFittedEstimator(Estimator[Dict[str, Any]]):
    """
    Base class for cross-fitted estimators (DRCPO, MRDR, etc).

    Handles common functionality:
    - Cross-validation setup
    - Weight computation and calibration
    - Parallel fold processing
    - Result aggregation
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
        judge_runner: Optional[Any] = None,
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

        # State
        self.n: int = 0
        self.K: int = sampler.K
        self._folds: Optional[List[np.ndarray]] = None
        self._full_logs_data: List[Dict[str, Any]] = []
        self._rewards_full: Optional[np.ndarray] = None
        self._features_full: Optional[np.ndarray] = None
        self._logp_behavior_full: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self._weight_stats: Optional[Dict[str, Any]] = None

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """Common fit logic for all cross-fitted estimators."""
        self._full_logs_data = logs
        self.n = len(logs)

        if self.n == 0:
            raise ValueError("Cannot fit estimator with empty logs")

        # Auto-select outcome model if needed
        if self._should_auto_select_models():
            self._auto_select_outcome_model()

        # Extract data
        self._rewards_full = np.array([log["reward"] for log in logs], dtype=float)
        self._logp_behavior_full = np.array([log["logp"] for log in logs], dtype=float)

        # Compute importance weights
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

        # Fit featurizer
        console.print("[bold blue]Featurizing data...[/bold blue]")
        self.featurizer.fit(logs)
        self._features_full = self.featurizer.transform(logs)

        # Setup cross-validation
        indices = np.arange(self.n)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        self._folds = np.array_split(indices, self.k)

    @abstractmethod
    def _process_fold(self, fold_indices: np.ndarray, fold_num: int = 0) -> np.ndarray:
        """Process a single fold. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_estimator_type(self) -> str:
        """Return estimator type for results."""
        pass

    def estimate(self) -> EstimationResult:
        """Common estimation logic."""
        if self._folds is None:
            raise RuntimeError("Must call fit() before estimate()")

        console.print(
            f"[bold blue]Running {self.k}-fold cross-validation...[/bold blue]"
        )

        # Process folds
        with ProgressMonitor() as progress:
            progress.add_task(
                "cv_folds", f"Processing {self.k} cross-validation folds", total=self.k
            )

            if self.n < 100 or self.n_jobs == 1 or self.k <= 2:
                # Sequential processing
                results = []
                for i, fold_indices in enumerate(self._folds):
                    fold_result = self._process_fold(fold_indices, i + 1)
                    results.append(fold_result)
                    progress.update("cv_folds", 1)
            else:
                # Parallel processing
                results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(self._process_fold)(fold_indices, i + 1)
                    for i, fold_indices in enumerate(self._folds)
                )
                progress.update("cv_folds", self.k)

        # Aggregate results
        eif_all = cast(np.ndarray, np.vstack(results))
        v_hat = np.mean(eif_all, axis=0)

        # Compute covariance
        if self.K == 1:
            Sigma_hat = np.array([[np.var(eif_all[:, 0], ddof=1) / self.n]])
        else:
            Sigma_hat = np.cov(eif_all, rowvar=False) / self.n

        se = np.sqrt(np.diag(Sigma_hat))

        # Create metadata
        metadata = EstimatorMetadata(
            estimator_type=self.get_estimator_type(),
            k_folds=self.k,
            stabilize_weights=self.stabilize_weights,
            bootstrap_available=True,
            calibrate_weights=self.calibrate_weights,
            calibrate_outcome=self.calibrate_outcome,
        )

        if self._weight_stats:
            metadata.ess_values = self._weight_stats["ess_values"]
            metadata.ess_percentage = self._weight_stats["ess_percentage"]

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            eif_components=eif_all,
            covariance_matrix=Sigma_hat,
            estimator_type=self.get_estimator_type(),
            n_policies=self.K,
            metadata=metadata.to_dict(),
        )

    def _should_auto_select_models(self) -> bool:
        """Check if we should auto-select outcome model."""
        from sklearn.linear_model import Ridge

        return self.outcome_model_cls is None or (
            self.outcome_model_cls is Ridge and not self.outcome_model_kwargs
        )

    def _auto_select_outcome_model(self) -> None:
        """Auto-select outcome model based on data size."""
        from sklearn.linear_model import Ridge

        self.outcome_model_cls, self.outcome_model_kwargs, base_featurizer = (
            auto_select(self.n)
        )

        if self.judge_runner is not None:
            from .auto_outcome import ScoreAugmentFeaturizer

            # Include variance information from unified judge system
            self.featurizer = ScoreAugmentFeaturizer(
                base_featurizer, include_variance=True
            )
        else:
            self.featurizer = base_featurizer
