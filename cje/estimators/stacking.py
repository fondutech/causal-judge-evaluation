"""Estimator stacking via influence function variance minimization.

This module implements stacking of DR-family estimators (DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO)
by forming an optimal convex combination that minimizes the variance of the
combined influence function.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from cje.estimators.base_estimator import BaseCJEEstimator
from cje.data.models import EstimationResult
from cje.data.precomputed_sampler import PrecomputedSampler

logger = logging.getLogger(__name__)


class StackedDREstimator(BaseCJEEstimator):
    """
    Stacks DR estimators via influence function variance minimization.

    This implements the estimator stacking approach from the CJE paper,
    forming an optimal convex combination of DR-family estimators
    by minimizing the empirical variance of the combined influence function.

    Key features:
    - Runs multiple DR estimators with shared resources (folds, fresh draws)
    - Optional IIC at component level for variance reduction (use_iic=True)
    - Computes optimal weights via exact QP for K≤6 or projection for larger K
    - Uses outer split for honest inference (default)
    - Robust covariance estimation via adaptive Ledoit-Wolf shrinkage
    - Parallel execution of component estimators

    Example:
        >>> sampler = PrecomputedSampler.from_jsonl("data.jsonl")
        >>> stacked = StackedDREstimator(sampler, use_iic=True)
        >>> result = stacked.fit_and_estimate()
        >>> print(f"Estimate: {result.estimates[0]:.3f} ± {result.standard_errors[0]:.3f}")
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        estimators: Optional[List[str]] = None,
        use_outer_split: bool = True,
        V_folds: int = 5,
        robust_cov: bool = True,
        shrinkage_intensity: Optional[float] = None,
        parallel: bool = True,
        min_weight: float = 0.0,
        fallback_on_failure: bool = True,
        seed: int = 42,
        use_iic: bool = False,  # Whether component estimators should use IIC
        oua_jackknife: bool = True,  # Oracle uncertainty augmentation for components
        **kwargs: Any,
    ):
        """Initialize the stacked estimator.

        Args:
            sampler: PrecomputedSampler with calibrated data
            estimators: List of estimator names to stack.
                Default: ["dr-cpo", "tmle", "mrdr"]
                Available: "dr-cpo", "tmle", "mrdr", "oc-dr-cpo", "tr-cpo", "tr-cpo-e"
                Note: "tr-cpo" uses raw W (vanilla), "tr-cpo-e" uses m̂(S) (efficient)
            use_outer_split: If True, use V-fold outer split for honest inference
            V_folds: Number of outer folds for honest stacking
            robust_cov: If True, use Ledoit-Wolf optimal shrinkage for covariance
            shrinkage_intensity: Manual override for shrinkage (0 to 1, None for automatic)
            parallel: If True, run component estimators in parallel
            min_weight: Minimum weight for any estimator (for stability)
            fallback_on_failure: If True, fall back to best single estimator on failure
            seed: Random seed for reproducibility
            use_iic: If True, component estimators use IIC for variance reduction
            oua_jackknife: If True, enable Oracle Uncertainty Augmentation for component estimators
            **kwargs: Additional arguments passed to base class
        """
        # Extract calibrator before passing to base class (base doesn't accept it)
        self.reward_calibrator = kwargs.pop("reward_calibrator", None)
        # Extract weight_mode for DR estimators
        self.weight_mode = kwargs.pop("weight_mode", "hajek")
        # Extract use_calibrated_weights to control SIMCal
        self.use_calibrated_weights = kwargs.pop("use_calibrated_weights", True)
        # Remove deprecated oracle_slice_config parameter (now using OUA jackknife)
        kwargs.pop("oracle_slice_config", None)  # Remove if present, ignore if not

        # BaseCJEEstimator only accepts specific params, not arbitrary kwargs
        # Oracle slice config removed - OUA jackknife handled per-estimator
        super().__init__(sampler)

        # Configuration
        # Default to a diverse set of estimators for robustness
        self.estimators = estimators or [
            "dr-cpo",
            "tmle",
            "mrdr",
            "oc-dr-cpo",
            "tr-cpo",
            "tr-cpo-e",
        ]
        self.use_outer_split = use_outer_split
        self.V_folds = V_folds
        self.robust_cov = robust_cov
        self.shrinkage_intensity = shrinkage_intensity
        self.parallel = parallel
        self.min_weight = min_weight
        self.fallback_on_failure = fallback_on_failure
        self.seed = seed
        self.use_iic = use_iic
        self.oua_jackknife = oua_jackknife  # Store OUA setting

        # Storage for results
        self.component_results: Dict[str, Optional[EstimationResult]] = {}
        self.component_estimators: Dict[str, Any] = (
            {}
        )  # Store estimators for weights/diagnostics
        self.weights_per_policy: Dict[str, np.ndarray] = {}
        self.stacking_diagnostics: Dict[str, Any] = {}
        self._fresh_draws: Dict[str, Any] = (
            {}
        )  # Store fresh draws to pass to components

        # Set up shared resources
        self._setup_shared_resources()

    def _setup_shared_resources(self) -> None:
        """Set up resources shared across all component estimators."""
        np.random.seed(self.seed)

        # Generate fold assignments once using unified system (all estimators use same folds)
        from ..data.folds import get_folds_for_dataset

        # Use the unified fold system if we have a real dataset
        if hasattr(self.sampler, "dataset"):
            n_folds = getattr(self.sampler.dataset, "metadata", {}).get("n_folds", 5)
            seed = getattr(self.sampler.dataset, "metadata", {}).get(
                "fold_seed", self.seed
            )
            self.shared_fold_ids = get_folds_for_dataset(
                self.sampler.dataset,
                n_folds=n_folds,
                seed=seed,
            )
        else:
            # Fallback for mock objects in tests
            try:
                n = len(self.sampler)
            except TypeError:
                n = (
                    self.sampler.n_valid_samples
                    if hasattr(self.sampler, "n_valid_samples")
                    else 100
                )
            # Simple assignment for mocks
            self.shared_fold_ids = np.arange(n) % 5

        # Note: Fresh draws are already in the sampler if available
        # Each estimator will detect and use them automatically

        logger.info(f"Set up shared resources for {len(self.estimators)} estimators")

    def add_fresh_draws(self, policy: str, fresh_draws: Any) -> None:
        """Store fresh draws to pass to component estimators.

        Args:
            policy: Target policy name
            fresh_draws: Fresh draw dataset for this policy
        """
        self._fresh_draws[policy] = fresh_draws
        logger.debug(f"Added fresh draws for policy {policy}")

    def fit(self) -> None:
        """Fit is a no-op for stacking (component estimators handle their own fitting)."""
        self._fitted = True

    def estimate(self) -> EstimationResult:
        """Run all component estimators and stack them optimally."""
        if not self._fitted:
            self.fit()

        # Step 1: Run all component estimators
        self._run_all_estimators()

        # Check for failures
        valid_estimators = self._get_valid_estimators()

        if len(valid_estimators) == 0:
            raise RuntimeError("All component estimators failed")

        if len(valid_estimators) == 1:
            # Only one valid estimator, pass through its results
            logger.warning(
                f"Only one valid estimator ({valid_estimators[0]}), using it directly"
            )
            return self._create_passthrough_result(valid_estimators[0])

        # Step 2: Stack estimates for each policy
        stacked_estimates = []
        stacked_ses = []
        stacked_ifs = {}

        # Track which components contributed to each policy's stack
        self._weights_names_per_policy: Dict[str, List[str]] = {}

        for policy_idx, policy in enumerate(self.sampler.target_policies):
            # Collect influence functions and track which estimators provided them
            IF_matrix, used_names = self._collect_influence_functions_with_names(
                policy, valid_estimators
            )

            if IF_matrix is None or IF_matrix.shape[1] == 0:
                # No valid IFs for this policy
                logger.warning(f"No valid influence functions for policy {policy}")
                stacked_estimates.append(np.nan)
                stacked_ses.append(np.nan)
                continue

            # Compute stacked influence function
            if self.use_outer_split:
                # Learn fold-specific weights but use averaged weights for consistency
                _, weights = self._stack_with_outer_split(IF_matrix, policy)
            else:
                weights = self._compute_optimal_weights(IF_matrix)

            # Use the same weights for both IF and point estimate (critical for alignment)
            stacked_if = IF_matrix @ weights
            self.weights_per_policy[policy] = weights
            self._weights_names_per_policy[policy] = used_names

            # Note: IIC is applied at the component level, not at the stacked level
            # This avoids double residualization and maintains proper alignment

            # Store influence function
            stacked_ifs[policy] = stacked_if
            self._influence_functions[policy] = stacked_if

            # Compute stacked estimate using ONLY components that provided IFs
            component_estimates = []
            for est_name in used_names:  # Use same components as IF_matrix
                result = self.component_results[est_name]
                if result:
                    component_estimates.append(result.estimates[policy_idx])

            if component_estimates:
                # Compute weighted average of component estimates
                # Note: Components already adjust their estimates if IIC is used,
                # so we don't need to subtract offsets here (avoids double adjustment)
                estimate = np.dot(weights, component_estimates)
            else:
                estimate = np.nan

            # Compute SE from the stacked influence function
            se = np.std(stacked_if, ddof=1) / np.sqrt(len(stacked_if))

            stacked_estimates.append(estimate)
            stacked_ses.append(se)

        # Step 3: Create result object first (needed for diagnostics)
        metadata = {
            "stacking_weights": self.weights_per_policy,
            "valid_estimators": valid_estimators,
            "failed_estimators": [
                e for e in self.estimators if e not in valid_estimators
            ],
            "used_outer_split": self.use_outer_split,
            "V_folds": self.V_folds if self.use_outer_split else None,
            "component_iic": self.use_iic,  # IIC applied at component level
        }

        # Get n_samples_used from one of the component estimators
        n_samples_used: Dict[str, int] = {}
        if valid_estimators and self.component_results[valid_estimators[0]] is not None:
            first_result = self.component_results[valid_estimators[0]]
            if first_result is not None:  # Type guard
                n_samples_used = first_result.n_samples_used
        else:
            # Fallback: use sampler info
            for policy in self.sampler.target_policies:
                n_samples_used[policy] = len(self.sampler)

        result = EstimationResult(
            estimates=np.array(stacked_estimates),
            standard_errors=np.array(stacked_ses),
            n_samples_used=n_samples_used,
            method=f"StackedDR({', '.join(valid_estimators)})",
            influence_functions=stacked_ifs,
            diagnostics=None,  # Will be set after building diagnostics
            metadata=metadata,
            robust_standard_errors=None,
            robust_confidence_intervals=None,
        )

        # Set _results before building diagnostics (needed for variance reduction calculation)
        self._results = result

        # Step 4: Build diagnostics (now that self._results is set)
        diagnostics = self._build_stacking_diagnostics(valid_estimators)

        # Add diagnostics to metadata
        result.metadata["stacking_diagnostics"] = diagnostics

        # Step 5: Build proper DRDiagnostics for the stacked result
        dr_diagnostics = self._build_stacked_dr_diagnostics(valid_estimators, result)
        if dr_diagnostics:
            result.diagnostics = dr_diagnostics

        # Step 6: Apply stacked OUA (Oracle Uncertainty Augmentation)
        self._apply_stacked_oua(result, valid_estimators)

        return result

    def fit_and_estimate(self) -> EstimationResult:
        """Convenience method to fit and estimate in one call."""
        self.fit()
        return self.estimate()

    def _run_all_estimators(self) -> None:
        """Run all component estimators, either in parallel or sequentially."""
        logger.info(f"Running {len(self.estimators)} component estimators")

        if self.parallel:
            with ThreadPoolExecutor(max_workers=len(self.estimators)) as executor:
                futures = {
                    executor.submit(self._run_single_estimator, name): name
                    for name in self.estimators
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        result = future.result()
                        self.component_results[name] = result
                        logger.info(f"Completed {name}")
                    except Exception as e:
                        logger.warning(f"Estimator {name} failed: {e}")
                        self.component_results[name] = None
        else:
            for name in self.estimators:
                try:
                    result = self._run_single_estimator(name)
                    self.component_results[name] = result
                    logger.info(f"Completed {name}")
                except Exception as e:
                    logger.warning(f"Estimator {name} failed: {e}")
                    self.component_results[name] = None

    def _run_single_estimator(self, name: str) -> EstimationResult:
        """Run a single component estimator with shared resources."""
        # Import here to avoid circular imports
        from cje.estimators.dr_base import DRCPOEstimator
        from cje.estimators.tmle import TMLEEstimator
        from cje.estimators.mrdr import MRDREstimator
        from cje.estimators.orthogonalized_calibrated_dr import (
            OrthogonalizedCalibratedDRCPO,
        )
        from cje.estimators.tr_cpo import TRCPOEstimator

        estimator_classes = {
            "dr-cpo": DRCPOEstimator,
            "tmle": TMLEEstimator,
            "mrdr": MRDREstimator,
            "oc-dr-cpo": OrthogonalizedCalibratedDRCPO,
            "tr-cpo": TRCPOEstimator,  # Raw weights version (vanilla)
            "tr-cpo-e": TRCPOEstimator,  # Efficient version with m̂(S)
        }

        if name not in estimator_classes:
            raise ValueError(f"Unknown estimator: {name}")

        estimator_class = estimator_classes[name]

        # Create estimator with shared fold assignments and calibrator
        # Note: We can't directly pass fold_ids to most estimators,
        # but they will use the same seed which helps
        # Pass reward_calibrator as a named parameter for DR estimators
        if name in ["dr-cpo", "tmle", "mrdr", "oc-dr-cpo", "tr-cpo", "tr-cpo-e"]:
            # Always pass reward_calibrator for outcome model (if available)
            # use_calibrated_weights controls SIMCal, independent of reward_calibrator

            # Build kwargs for estimator
            estimator_kwargs = {
                "reward_calibrator": self.reward_calibrator,
                "weight_mode": self.weight_mode,
                "oua_jackknife": self.oua_jackknife,  # Pass OUA jackknife setting
                "use_iic": self.use_iic,  # Enable IIC for component estimators
            }

            # Only pass use_calibrated_weights to estimators that support it
            # TR-CPO variants ignore this parameter (always use raw weights)
            if name not in ["tr-cpo", "tr-cpo-e"]:
                estimator_kwargs["use_calibrated_weights"] = self.use_calibrated_weights

            # Configure TR-CPO variants
            if name == "tr-cpo":
                # Vanilla TR-CPO uses raw W (theoretical form, high variance)
                estimator_kwargs["use_efficient_tr"] = False
            elif name == "tr-cpo-e":
                # Efficient TR-CPO uses m̂(S) for stability
                estimator_kwargs["use_efficient_tr"] = True

            estimator = estimator_class(self.sampler, **estimator_kwargs)

            # Pass shared fold IDs if the estimator supports it
            if hasattr(estimator, "set_fold_ids"):
                estimator.set_fold_ids(self.shared_fold_ids)
            elif hasattr(self, "shared_fold_ids"):
                # Try to set it directly if the estimator has the attribute
                try:
                    estimator.fold_ids = self.shared_fold_ids
                    logger.debug(f"Set fold_ids for {name}")
                except AttributeError:
                    logger.debug(
                        f"{name} does not accept fold_ids; using its internal split"
                    )
        else:
            # For non-DR estimators (shouldn't happen with default config)
            estimator = estimator_class(self.sampler)

            # Try to pass fold IDs even for non-DR estimators
            if hasattr(estimator, "set_fold_ids"):
                estimator.set_fold_ids(self.shared_fold_ids)

        # Add fresh draws if available
        if self._fresh_draws:
            for policy, fresh_draws in self._fresh_draws.items():
                estimator.add_fresh_draws(policy, fresh_draws)
                logger.debug(f"Added fresh draws for {policy} to {name}")

        # Store the estimator for later access
        self.component_estimators[name] = estimator

        # Run estimation
        result: EstimationResult = estimator.fit_and_estimate()
        return result

    def _get_valid_estimators(self) -> List[str]:
        """Get list of estimators that ran successfully."""
        valid = []
        for name in self.estimators:
            result = self.component_results.get(name)
            if result is not None and not np.all(np.isnan(result.estimates)):
                valid.append(name)
        return valid

    def _collect_influence_functions(
        self, policy: str, valid_estimators: List[str]
    ) -> Optional[np.ndarray]:
        """Collect influence functions from valid estimators for a given policy.

        Returns:
            IF_matrix: n x K matrix where K is the number of valid estimators
        """
        IF_matrix, _ = self._collect_influence_functions_with_names(
            policy, valid_estimators
        )
        return IF_matrix

    def _collect_influence_functions_with_names(
        self, policy: str, valid_estimators: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Collect influence functions and track which estimators provided them.

        This ensures alignment between weights and estimates by tracking exactly
        which components contributed IFs.

        Returns:
            IF_matrix: n x K matrix where K is the number of estimators with IFs
            used_names: List of estimator names that provided IFs (length K)
        """
        ifs: List[np.ndarray] = []
        used_names: List[str] = []

        for name in valid_estimators:
            result = self.component_results[name]
            if result and result.influence_functions:
                if_for_policy = result.influence_functions.get(policy)
                if if_for_policy is not None:
                    ifs.append(if_for_policy)
                    used_names.append(name)

        if not ifs:
            return (None, [])

        # Stack into matrix (each column is one estimator's IF)
        return (np.column_stack(ifs), used_names)

    def _compute_optimal_weights(self, IF_matrix: np.ndarray) -> np.ndarray:
        """Compute optimal stacking weights by minimizing IF variance.

        Solves: min_α α^T Σ α  s.t.  α ≥ 0, 1^T α = 1

        Args:
            IF_matrix: n x K matrix of influence functions

        Returns:
            weights: K-dimensional weight vector
        """
        # Clean IF matrix: remove rows with any NaN/Inf
        mask = np.all(np.isfinite(IF_matrix), axis=1)
        IF_clean = IF_matrix[mask]

        # Log if we dropped rows
        n_dropped = IF_matrix.shape[0] - IF_clean.shape[0]
        if n_dropped > 0:
            logger.debug(f"Dropped {n_dropped} rows with NaN/Inf from IF matrix")

        # If too few valid rows, fall back to uniform weights
        if IF_clean.shape[0] < 2:
            logger.warning("Too few valid rows in IF matrix, using uniform weights")
            return np.ones(IF_matrix.shape[1]) / IF_matrix.shape[1]

        K = IF_clean.shape[1]

        # Compute covariance matrix
        if self.robust_cov:
            Sigma = self._ledoit_wolf_covariance(IF_clean)
        else:
            # Center the IFs first
            centered_IF = IF_clean - IF_clean.mean(axis=0, keepdims=True)
            Sigma = np.cov(centered_IF.T)

        # Add small ridge for numerical stability
        Sigma = Sigma + 1e-8 * np.eye(K)

        # For small K, use exact QP solver; otherwise use projection
        if K <= 6:
            weights = self._solve_qp_exact(Sigma)
        else:
            weights = self._solve_qp_projection(Sigma)

        # Apply minimum weight constraint if specified
        if self.min_weight > 0:
            weights = np.maximum(weights, self.min_weight)
            weights = weights / weights.sum()

        return weights

    def _solve_qp_projection(self, Sigma: np.ndarray) -> np.ndarray:
        """Solve QP via closed-form + projection (approximate)."""
        K = Sigma.shape[0]
        ones = np.ones(K)

        try:
            # Solve Σw = 1
            weights = np.linalg.solve(Sigma, ones)
            # Normalize to sum to 1
            weights = weights / weights.sum()
            # Project to simplex
            weights = self._project_to_simplex(weights)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using uniform weights")
            weights = ones / K

        return weights

    def _solve_qp_exact(self, Sigma: np.ndarray) -> np.ndarray:
        """Solve quadratic program exactly.

        Minimize: 0.5 * α^T Σ α
        Subject to: 1^T α = 1, α ≥ 0
        """
        K = Sigma.shape[0]

        # Simple active-set method for small K
        best_weights = None
        best_value = float("inf")

        # Try all possible active sets (2^K combinations for small K)
        for active_mask in range(1, 2**K):
            active = [(active_mask >> i) & 1 for i in range(K)]
            active_idx = [i for i in range(K) if active[i]]

            if not active_idx:
                continue

            # Solve on active set
            Sigma_active = Sigma[np.ix_(active_idx, active_idx)]
            ones_active = np.ones(len(active_idx))

            try:
                # Solve equality-constrained problem
                w_active = np.linalg.solve(Sigma_active, ones_active)
                w_active = w_active / w_active.sum()

                # Check if feasible (all non-negative)
                if np.all(w_active >= -1e-10):
                    # Construct full weight vector
                    w_full = np.zeros(K)
                    for i, idx in enumerate(active_idx):
                        w_full[idx] = max(0, w_active[i])
                    w_full = w_full / w_full.sum()

                    # Compute objective value
                    obj_val = 0.5 * w_full @ Sigma @ w_full

                    if obj_val < best_value:
                        best_value = obj_val
                        best_weights = w_full
            except np.linalg.LinAlgError:
                continue

        if best_weights is not None:
            return best_weights
        else:
            # Fallback to uniform
            return np.ones(K) / K

    def _project_to_simplex(self, v: np.ndarray) -> np.ndarray:
        """Project vector v onto the probability simplex."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        ind = np.arange(1, n + 1)
        cond = u - cssv / ind > 0
        rho = np.nonzero(cond)[0][-1] if np.any(cond) else n - 1
        theta = cssv[rho] / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w / w.sum() if w.sum() > 0 else np.ones(n) / n

    def _ledoit_wolf_covariance(self, IF_matrix: np.ndarray) -> np.ndarray:
        """Compute Ledoit-Wolf optimal shrinkage covariance estimator.

        This uses sklearn's implementation of the Ledoit-Wolf (2004) estimator that
        adaptively chooses the shrinkage intensity to minimize expected squared error.

        The target is a diagonal matrix with sample variances.
        Reference: "A well-conditioned estimator for large-dimensional covariance matrices"
        """
        from sklearn.covariance import ledoit_wolf

        # Center the data
        X = IF_matrix - IF_matrix.mean(axis=0, keepdims=True)  # n x K

        # Compute Ledoit-Wolf shrunk covariance
        # ledoit_wolf returns (covariance, shrinkage_value)
        Sigma, shrinkage = ledoit_wolf(X)
        Sigma = np.asarray(Sigma)  # Ensure it's ndarray for mypy

        # Store shrinkage for diagnostics
        self._last_shrinkage = float(shrinkage)

        # Allow manual override if user explicitly set a value
        if self.shrinkage_intensity is not None:
            # Recompute with manual shrinkage
            shrinkage = float(np.clip(self.shrinkage_intensity, 0.0, 1.0))
            S = np.cov(X.T, bias=False)  # Sample covariance
            target = np.diag(np.diag(S))  # Diagonal target
            Sigma = (1 - shrinkage) * S + shrinkage * target
            self._last_shrinkage = shrinkage

        # Add small ridge for numerical stability
        # Cast to ndarray to satisfy mypy
        Sigma_array: np.ndarray = np.asarray(Sigma)
        result: np.ndarray = Sigma_array + 1e-8 * np.eye(Sigma_array.shape[0])
        return result

    def _stack_with_outer_split(
        self, IF_matrix: np.ndarray, policy: str
    ) -> Tuple[None, np.ndarray]:
        """Stack with V-fold outer split for honest weight learning.

        Uses outer split to learn weights honestly (avoiding overfitting),
        then returns averaged weights for consistent estimator/IF.

        Returns:
            None: Placeholder for compatibility
            avg_weights: Average weights across folds
        """
        n = IF_matrix.shape[0]
        fold_weights = []

        # Create V folds deterministically using seed
        indices = np.arange(n)
        rng = np.random.default_rng(self.seed if hasattr(self, "seed") else 42)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.V_folds)

        for test_idx in folds:
            # Training indices are all except test fold
            train_idx = np.setdiff1d(indices, test_idx)

            if len(train_idx) < 2:
                continue

            # Learn weights on training folds
            train_IF = IF_matrix[train_idx]
            weights = self._compute_optimal_weights(train_IF)
            fold_weights.append(weights)

        # Average weights across folds
        avg_weights = (
            np.mean(fold_weights, axis=0)
            if fold_weights
            else np.ones(IF_matrix.shape[1]) / IF_matrix.shape[1]
        )

        return None, avg_weights

    def _create_passthrough_result(self, estimator_name: str) -> EstimationResult:
        """Create a result that passes through a single estimator's results."""
        result = self.component_results[estimator_name]

        if result is None:
            raise ValueError(
                f"Cannot create passthrough for failed estimator {estimator_name}"
            )

        # Add metadata about the passthrough
        metadata = result.metadata.copy() if result.metadata else {}
        metadata.update(
            {
                "stacking_fallback": True,
                "reason": "Only one valid estimator",
                "selected_estimator": estimator_name,
                "attempted_estimators": self.estimators,
            }
        )

        return EstimationResult(
            estimates=result.estimates,
            standard_errors=result.standard_errors,
            n_samples_used=result.n_samples_used,
            method=f"StackedDR(fallback->{estimator_name})",
            influence_functions=result.influence_functions,
            diagnostics=result.diagnostics,
            metadata=metadata,
            robust_standard_errors=None,
            robust_confidence_intervals=None,
        )

    def _build_stacking_diagnostics(
        self, valid_estimators: List[str]
    ) -> Dict[str, Any]:
        """Build comprehensive diagnostics for the stacking procedure."""
        diagnostics: Dict[str, Any] = {
            "estimator_type": "StackedDR",
            "valid_estimators": valid_estimators,
            "failed_estimators": [
                e for e in self.estimators if e not in valid_estimators
            ],
            "n_components": len(valid_estimators),
            "used_outer_split": self.use_outer_split,
        }

        # Add weight diagnostics per policy
        diagnostics["weights_per_policy"] = {}
        for policy, weights in self.weights_per_policy.items():
            diagnostics["weights_per_policy"][policy] = {
                name: float(w) for name, w in zip(valid_estimators, weights)
            }

        # Add variance reduction metrics
        if len(valid_estimators) > 1:
            diagnostics["variance_reduction"] = self._compute_variance_reduction(
                valid_estimators
            )

        # Add IF correlation matrix for first policy (as example)
        if self.sampler.target_policies:
            first_policy = self.sampler.target_policies[0]
            IF_matrix = self._collect_influence_functions(
                first_policy, valid_estimators
            )
            if IF_matrix is not None and IF_matrix.shape[1] > 1:
                # Compute correlation matrix
                centered_IF = IF_matrix - IF_matrix.mean(axis=0, keepdims=True)
                corr_matrix = np.corrcoef(centered_IF.T)
                diagnostics["if_correlation_matrix"] = corr_matrix.tolist()

        return diagnostics

    def _compute_variance_reduction(self, valid_estimators: List[str]) -> Dict:
        """Compute variance reduction compared to individual estimators."""
        reduction = {}

        for policy_idx, policy in enumerate(self.sampler.target_policies):
            if policy not in self.weights_per_policy:
                continue

            # Get stacked SE
            if self._results:
                stacked_se = self._results.standard_errors[policy_idx]

                # Compare to each component
                for name in valid_estimators:
                    result = self.component_results.get(name)
                    if result and not np.isnan(result.standard_errors[policy_idx]):
                        component_se = result.standard_errors[policy_idx]
                        pct_reduction = 100 * (1 - stacked_se**2 / component_se**2)
                        reduction[f"{policy}_vs_{name}"] = float(pct_reduction)

        return reduction

    def get_weights(self, policy: str) -> Optional[np.ndarray]:
        """Get importance weights for a given policy.

        For stacked estimator, returns the weights from the first successful component
        (all DR estimators use the same IPS weights).

        Args:
            policy: Target policy name

        Returns:
            Importance weights or None if not available
        """
        # Try to get weights from first successful component estimator
        # (all DR estimators use the same IPS weights)
        for name in ["dr-cpo", "tmle", "mrdr", "tr-cpo", "tr-cpo-e", "oc-dr-cpo"]:
            if name in self.component_estimators:
                estimator = self.component_estimators[name]
                if estimator:
                    # DR estimators store weights in their internal ips_estimator
                    if hasattr(estimator, "ips_estimator") and hasattr(
                        estimator.ips_estimator, "get_weights"
                    ):
                        weights = estimator.ips_estimator.get_weights(policy)
                        return np.asarray(weights) if weights is not None else None
                    elif hasattr(estimator, "get_weights"):
                        weights = estimator.get_weights(policy)
                        return np.asarray(weights) if weights is not None else None

        return None

    def get_diagnostics(self) -> Optional[Any]:
        """Get diagnostics from the stacked estimator.

        Returns the stacked diagnostics if available, otherwise from first component.

        Returns:
            Diagnostics object or None
        """
        # Return stacked diagnostics if we have them
        if hasattr(self, "_results") and self._results and self._results.diagnostics:
            return self._results.diagnostics

        # Otherwise try to get diagnostics from the first successful DR component
        for name in ["dr-cpo", "tmle", "mrdr", "tr-cpo", "tr-cpo-e", "oc-dr-cpo"]:
            if name in self.component_estimators:
                estimator = self.component_estimators[name]
                if estimator and hasattr(estimator, "get_diagnostics"):
                    diag = estimator.get_diagnostics()
                    if diag is not None:
                        return diag

        # If no component has diagnostics, return None
        return None

    def _build_stacked_dr_diagnostics(
        self, valid_estimators: List[str], result: EstimationResult
    ) -> Optional[Any]:
        """Build proper DR diagnostics using shared weight information.

        Priority: Get weight diagnostics from components that respect use_calibrated_weights.
        TR-CPO variants always use raw weights, so only use them as fallback.

        Args:
            valid_estimators: List of successfully run estimator names
            result: The stacked estimation result

        Returns:
            DRDiagnostics object or None if unable to build
        """
        from ..diagnostics import DRDiagnostics, Status

        # Priority order for weight diagnostics based on alignment with stack settings
        # TR-CPO variants always use raw weights regardless of settings
        non_tr_estimators = ["dr-cpo", "oc-dr-cpo", "mrdr", "tmle"]
        tr_estimators = ["tr-cpo", "tr-cpo-e"]

        weight_diag = None
        source_estimator = None

        # First try non-TR estimators (they respect use_calibrated_weights)
        for name in non_tr_estimators:
            if name in valid_estimators and name in self.component_estimators:
                comp = self.component_estimators[name]
                if hasattr(comp, "get_weight_diagnostics"):
                    weight_diag = comp.get_weight_diagnostics()
                    if weight_diag:
                        source_estimator = name
                        logger.debug(f"Using weight diagnostics from {name}")
                        break

        # Fall back to TR-CPO if needed (but warn if settings mismatch)
        if not weight_diag:
            for name in tr_estimators:
                if name in valid_estimators and name in self.component_estimators:
                    comp = self.component_estimators[name]
                    if hasattr(comp, "get_weight_diagnostics"):
                        weight_diag = comp.get_weight_diagnostics()
                        if weight_diag:
                            source_estimator = name
                            if self.use_calibrated_weights:
                                logger.warning(
                                    f"Using weight diagnostics from {name} which uses raw weights, "
                                    f"but stack requested calibrated weights"
                                )
                            break

        if not weight_diag:
            logger.debug("No weight diagnostics available from components")
            return None

        # Build DRDiagnostics with stacked estimates but weight fields from appropriate component
        policies = list(self.sampler.target_policies)
        estimates_dict = {
            p: float(e) for p, e in zip(policies, result.estimates) if not np.isnan(e)
        }
        se_dict = {
            p: float(se)
            for p, se in zip(policies, result.standard_errors)
            if not np.isnan(se)
        }

        # Get n_folds from a component (they should all be the same)
        n_folds = 5  # default
        for name in valid_estimators:
            if name in self.component_estimators:
                comp = self.component_estimators[name]
                if hasattr(comp, "n_folds"):
                    n_folds = comp.n_folds
                    break

        return DRDiagnostics(
            estimator_type="StackedDR",
            method=f"StackedDR({', '.join(valid_estimators)})",
            n_samples_total=weight_diag.n_samples_total,
            n_samples_valid=weight_diag.n_samples_valid,
            n_policies=len(policies),
            policies=policies,
            estimates=estimates_dict,
            standard_errors=se_dict,
            n_samples_used=result.n_samples_used,
            # Weight fields from the appropriate component
            weight_ess=weight_diag.weight_ess,
            weight_status=weight_diag.weight_status,
            ess_per_policy=weight_diag.ess_per_policy,
            max_weight_per_policy=weight_diag.max_weight_per_policy,
            weight_tail_ratio_per_policy=getattr(
                weight_diag, "weight_tail_ratio_per_policy", {}
            ),
            # Calibration fields if available
            calibration_rmse=weight_diag.calibration_rmse,
            calibration_r2=weight_diag.calibration_r2,
            calibration_coverage=getattr(weight_diag, "calibration_coverage", None),
            n_oracle_labels=weight_diag.n_oracle_labels,
            # DR-specific fields
            dr_cross_fitted=True,
            dr_n_folds=n_folds,
            # Stacked estimator doesn't have single outcome model stats
            outcome_r2_range=(0.0, 0.0),
            outcome_rmse_mean=0.0,
            worst_if_tail_ratio=0.0,
            dr_diagnostics_per_policy={},
            dm_ips_decompositions={},
            orthogonality_scores={},
            influence_functions=self._influence_functions,
        )

    def _apply_stacked_oua(
        self, result: EstimationResult, valid_estimators: List[str]
    ) -> None:
        """Apply OUA by linearly combining component jackknife estimates.

        For stacked estimator ψ_stack = Σ α_k ψ_k with fixed weights α_k,
        the jackknife estimates combine linearly: ψ_stack^(-f) = Σ α_k ψ_k^(-f)

        Oracle variance is then: var_oracle = (K-1)/K * Var_f(ψ_stack^(-f))

        Args:
            result: EstimationResult to augment with robust standard errors
            valid_estimators: List of successfully run estimator names
        """
        if not self.oua_jackknife:
            logger.debug("OUA jackknife disabled for stacked estimator")
            return

        try:
            oua_ses = []
            var_oracle_map = {}
            jk_counts = {}
            jk_contributors: Dict[str, List[str]] = {}

            policies = list(self.sampler.target_policies)

            for policy_idx, policy in enumerate(policies):
                weights = self.weights_per_policy.get(policy)
                # Get the names of components that contributed to this policy
                used_names = self._weights_names_per_policy.get(policy, [])

                if weights is None or not used_names or len(weights) != len(used_names):
                    # No weights for this policy, keep base SE
                    oua_ses.append(float(result.standard_errors[policy_idx]))
                    var_oracle_map[policy] = 0.0
                    jk_counts[policy] = 0
                    continue

                # Collect component jackknife arrays (aligned by fold)
                component_jacks = []
                contributors = []

                # Use the same component names that were used for stacking
                for est_idx, name in enumerate(used_names):
                    weight = weights[est_idx]
                    if weight <= 0:
                        continue  # Skip zero-weight components

                    comp = self.component_estimators.get(name)
                    if not comp:
                        continue

                    # Try to get jackknife estimates from component
                    if hasattr(comp, "get_oracle_jackknife"):
                        jack_array = comp.get_oracle_jackknife(policy)
                        if jack_array is not None and len(jack_array) > 0:
                            component_jacks.append((weight, jack_array))
                            contributors.append(name)

                if not component_jacks:
                    # No OUA available from components, keep base SE
                    oua_ses.append(float(result.standard_errors[policy_idx]))
                    var_oracle_map[policy] = 0.0
                    jk_counts[policy] = 0
                    jk_contributors[policy] = []
                    logger.debug(f"No OUA jackknife available for {policy}")
                    continue

                # Check fold alignment (all should have same K)
                K_values = [len(jack) for _, jack in component_jacks]
                if len(set(K_values)) > 1:
                    logger.warning(
                        f"Components have misaligned folds for {policy}: {K_values}, using min"
                    )
                K = min(K_values)

                if K < 2:
                    # Not enough folds for jackknife
                    oua_ses.append(float(result.standard_errors[policy_idx]))
                    var_oracle_map[policy] = 0.0
                    jk_counts[policy] = K
                    jk_contributors[policy] = contributors
                    continue

                # Form stacked jackknife by linear combination
                stacked_jack = np.zeros(K, dtype=float)
                for weight, jack_array in component_jacks:
                    # Truncate to minimum K if needed
                    stacked_jack += weight * jack_array[:K]

                # Compute oracle variance: (K-1)/K * Var(jackknife estimates)
                psi_bar = float(np.mean(stacked_jack))
                var_oracle = (K - 1) / K * float(np.mean((stacked_jack - psi_bar) ** 2))

                # Combine with base SE
                base_se = float(result.standard_errors[policy_idx])
                robust_se = float(np.sqrt(base_se**2 + var_oracle))

                oua_ses.append(robust_se)
                var_oracle_map[policy] = var_oracle
                jk_counts[policy] = K
                jk_contributors[policy] = contributors

                logger.debug(
                    f"Stacked OUA for {policy}: base_se={base_se:.4f}, "
                    f"var_oracle={var_oracle:.4f}, robust_se={robust_se:.4f}, "
                    f"K={K}, contributors={contributors}"
                )

            # Store robust standard errors
            result.robust_standard_errors = np.array(oua_ses, dtype=float)

            # Add OUA metadata
            if not hasattr(result, "metadata") or result.metadata is None:
                result.metadata = {}

            result.metadata.setdefault("oua", {})
            result.metadata["oua"].update(
                {
                    "source": "stacked-linear-combo",
                    "var_oracle_per_policy": var_oracle_map,
                    "jackknife_counts": jk_counts,
                    "contributors_per_policy": jk_contributors,
                }
            )

        except Exception as e:
            logger.error(f"Stacked OUA jackknife failed: {e}", exc_info=True)
            # On failure, leave robust_standard_errors as None
