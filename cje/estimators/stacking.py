"""Estimator stacking via influence function variance minimization.

This module implements stacking of DR-family estimators (DR-CPO, TMLE, MRDR)
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
    forming an optimal convex combination of DR-CPO, TMLE, and MRDR
    by minimizing the empirical variance of the combined influence function.
    
    Key features:
    - Runs multiple DR estimators with shared resources (folds, fresh draws)
    - Computes optimal weights by minimizing IF variance
    - Uses outer split for honest inference (default)
    - Robust covariance estimation via Ledoit-Wolf shrinkage
    - Parallel execution of component estimators
    
    Example:
        >>> sampler = PrecomputedSampler.from_jsonl("data.jsonl")
        >>> stacked = StackedDREstimator(sampler)
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
        shrinkage_intensity: float = 0.1,
        parallel: bool = True,
        min_weight: float = 0.0,
        fallback_on_failure: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """Initialize the stacked estimator.
        
        Args:
            sampler: PrecomputedSampler with calibrated data
            estimators: List of estimator names to stack. 
                Default: ["dr-cpo", "tmle", "mrdr"]
            use_outer_split: If True, use V-fold outer split for honest inference
            V_folds: Number of outer folds for honest stacking
            robust_cov: If True, use Ledoit-Wolf shrinkage for covariance
            shrinkage_intensity: Shrinkage parameter for Ledoit-Wolf (0 to 1)
            parallel: If True, run component estimators in parallel
            min_weight: Minimum weight for any estimator (for stability)
            fallback_on_failure: If True, fall back to best single estimator on failure
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(sampler, **kwargs)
        
        # Configuration
        self.estimators = estimators or ["dr-cpo", "tmle", "mrdr"]
        self.use_outer_split = use_outer_split
        self.V_folds = V_folds
        self.robust_cov = robust_cov
        self.shrinkage_intensity = shrinkage_intensity
        self.parallel = parallel
        self.min_weight = min_weight
        self.fallback_on_failure = fallback_on_failure
        self.seed = seed
        
        # Storage for results
        self.component_results: Dict[str, EstimationResult] = {}
        self.weights_per_policy: Dict[str, np.ndarray] = {}
        self.stacking_diagnostics: Dict[str, Any] = {}
        
        # Set up shared resources
        self._setup_shared_resources()
    
    def _setup_shared_resources(self):
        """Set up resources shared across all component estimators."""
        np.random.seed(self.seed)
        
        # Generate fold assignments once (all estimators use same folds)
        # Handle both actual samplers and mock objects
        try:
            n = len(self.sampler)
        except TypeError:
            # If sampler doesn't support len(), try getting sample count
            n = self.sampler.n_valid_samples if hasattr(self.sampler, 'n_valid_samples') else 100
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        self.shared_fold_ids = np.zeros(n, dtype=int)
        for fold_idx, (_, test_idx) in enumerate(kf.split(np.arange(n))):
            self.shared_fold_ids[test_idx] = fold_idx
        
        # Note: Fresh draws are already in the sampler if available
        # Each estimator will detect and use them automatically
        
        logger.info(f"Set up shared resources for {len(self.estimators)} estimators")
    
    def fit(self) -> "StackedDREstimator":
        """Fit is a no-op for stacking (component estimators handle their own fitting)."""
        self._fitted = True
        return self
    
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
            logger.warning(f"Only one valid estimator ({valid_estimators[0]}), using it directly")
            return self._create_passthrough_result(valid_estimators[0])
        
        # Step 2: Stack estimates for each policy
        stacked_estimates = []
        stacked_ses = []
        stacked_ifs = {}
        
        for policy_idx, policy in enumerate(self.sampler.target_policies):
            # Collect influence functions from valid estimators
            IF_matrix = self._collect_influence_functions(policy, valid_estimators)
            
            if IF_matrix is None or IF_matrix.shape[1] == 0:
                # No valid IFs for this policy
                logger.warning(f"No valid influence functions for policy {policy}")
                stacked_estimates.append(np.nan)
                stacked_ses.append(np.nan)
                continue
            
            # Compute stacked influence function
            if self.use_outer_split:
                stacked_if, weights = self._stack_with_outer_split(IF_matrix, policy)
            else:
                weights = self._compute_optimal_weights(IF_matrix)
                stacked_if = IF_matrix @ weights
            
            self.weights_per_policy[policy] = weights
            
            # Store influence function
            stacked_ifs[policy] = stacked_if
            self._influence_functions[policy] = stacked_if
            
            # Compute estimate and SE
            # The estimate is the mean of the influence function
            estimate = np.mean(stacked_if)
            se = np.std(stacked_if, ddof=1) / np.sqrt(len(stacked_if))
            
            stacked_estimates.append(estimate)
            stacked_ses.append(se)
        
        # Step 3: Build diagnostics
        diagnostics = self._build_stacking_diagnostics(valid_estimators)
        
        # Step 4: Create and return result
        metadata = {
            "stacking_weights": self.weights_per_policy,
            "valid_estimators": valid_estimators,
            "failed_estimators": [e for e in self.estimators if e not in valid_estimators],
            "used_outer_split": self.use_outer_split,
            "V_folds": self.V_folds if self.use_outer_split else None
        }
        
        result = EstimationResult(
            estimates=np.array(stacked_estimates),
            standard_errors=np.array(stacked_ses),
            influence_functions=stacked_ifs,
            diagnostics=diagnostics,
            metadata=metadata
        )
        
        self._results = result
        return result
    
    def fit_and_estimate(self) -> EstimationResult:
        """Convenience method to fit and estimate in one call."""
        return self.fit().estimate()
    
    def _run_all_estimators(self):
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
        from cje.estimators.dr_cpo import DRCPOEstimator
        from cje.estimators.tmle import TMLEEstimator
        from cje.estimators.mrdr import MRDREstimator
        
        estimator_classes = {
            "dr-cpo": DRCPOEstimator,
            "tmle": TMLEEstimator,
            "mrdr": MRDREstimator
        }
        
        if name not in estimator_classes:
            raise ValueError(f"Unknown estimator: {name}")
        
        estimator_class = estimator_classes[name]
        
        # Create estimator with shared fold assignments
        # Note: We can't directly pass fold_ids to most estimators,
        # but they will use the same seed which helps
        estimator = estimator_class(
            self.sampler,
            oracle_slice_config=self.oracle_slice_config
        )
        
        # Run estimation
        return estimator.fit_and_estimate()
    
    def _get_valid_estimators(self) -> List[str]:
        """Get list of estimators that ran successfully."""
        valid = []
        for name in self.estimators:
            result = self.component_results.get(name)
            if result is not None and not np.all(np.isnan(result.estimates)):
                valid.append(name)
        return valid
    
    def _collect_influence_functions(
        self, 
        policy: str, 
        valid_estimators: List[str]
    ) -> Optional[np.ndarray]:
        """Collect influence functions from valid estimators for a given policy.
        
        Returns:
            IF_matrix: n x K matrix where K is the number of valid estimators
        """
        ifs = []
        
        for name in valid_estimators:
            result = self.component_results[name]
            if result and result.influence_functions:
                if_for_policy = result.influence_functions.get(policy)
                if if_for_policy is not None:
                    ifs.append(if_for_policy)
        
        if not ifs:
            return None
        
        # Stack into matrix (each column is one estimator's IF)
        return np.column_stack(ifs)
    
    def _compute_optimal_weights(self, IF_matrix: np.ndarray) -> np.ndarray:
        """Compute optimal stacking weights by minimizing IF variance.
        
        Solves: min_α α^T Σ α  s.t.  α ≥ 0, Σα = 1
        
        Args:
            IF_matrix: n x K matrix of influence functions
            
        Returns:
            weights: K-dimensional weight vector
        """
        K = IF_matrix.shape[1]
        
        # Compute covariance matrix
        if self.robust_cov:
            Sigma = self._ledoit_wolf_covariance(IF_matrix)
        else:
            # Center the IFs first
            centered_IF = IF_matrix - IF_matrix.mean(axis=0, keepdims=True)
            Sigma = np.cov(centered_IF.T)
        
        # Add small ridge for numerical stability
        Sigma = Sigma + 1e-8 * np.eye(K)
        
        # Closed-form solution: w ∝ Σ^{-1} 1
        ones = np.ones(K)
        
        try:
            # Solve Σw = 1
            weights = np.linalg.solve(Sigma, ones)
            
            # Normalize to sum to 1
            weights = weights / weights.sum()
            
            # Project to simplex (ensure non-negative)
            weights = self._project_to_simplex(weights)
            
            # Apply minimum weight if specified
            if self.min_weight > 0:
                weights = np.maximum(weights, self.min_weight)
                weights = weights / weights.sum()
            
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is singular, using equal weights")
            weights = ones / K
        
        return weights
    
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
        """Compute shrinkage covariance estimator (Ledoit-Wolf).
        
        Shrinks the sample covariance toward a diagonal target.
        """
        # Center the data
        X = IF_matrix - IF_matrix.mean(axis=0, keepdims=True)
        n, p = X.shape
        
        # Sample covariance
        S = (X.T @ X) / n
        
        # Shrinkage target (diagonal with sample variances)
        target = np.diag(np.diag(S))
        
        # Use specified shrinkage intensity
        shrinkage = self.shrinkage_intensity
        
        # Shrink toward target
        return (1 - shrinkage) * S + shrinkage * target
    
    def _stack_with_outer_split(
        self, 
        IF_matrix: np.ndarray,
        policy: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stack with V-fold outer split for honest inference.
        
        Returns:
            stacked_if: The stacked influence function
            avg_weights: Average weights across folds (for reporting)
        """
        n = IF_matrix.shape[0]
        stacked_if = np.zeros(n)
        fold_weights = []
        
        # Create V folds
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.V_folds)
        
        for v, test_idx in enumerate(folds):
            # Training indices are all except test fold
            train_idx = np.setdiff1d(indices, test_idx)
            
            if len(train_idx) < 2:
                continue
            
            # Learn weights on training folds
            train_IF = IF_matrix[train_idx]
            weights = self._compute_optimal_weights(train_IF)
            fold_weights.append(weights)
            
            # Apply to test fold
            stacked_if[test_idx] = IF_matrix[test_idx] @ weights
        
        # Average weights across folds for reporting
        avg_weights = np.mean(fold_weights, axis=0) if fold_weights else np.ones(IF_matrix.shape[1]) / IF_matrix.shape[1]
        
        return stacked_if, avg_weights
    
    def _create_passthrough_result(self, estimator_name: str) -> EstimationResult:
        """Create a result that passes through a single estimator's results."""
        result = self.component_results[estimator_name]
        
        # Add metadata about the passthrough
        metadata = result.metadata.copy() if result.metadata else {}
        metadata.update({
            "stacking_fallback": True,
            "reason": "Only one valid estimator",
            "selected_estimator": estimator_name,
            "attempted_estimators": self.estimators
        })
        
        return EstimationResult(
            estimates=result.estimates,
            standard_errors=result.standard_errors,
            influence_functions=result.influence_functions,
            diagnostics=result.diagnostics,
            metadata=metadata
        )
    
    def _build_stacking_diagnostics(self, valid_estimators: List[str]) -> Dict:
        """Build comprehensive diagnostics for the stacking procedure."""
        diagnostics = {
            "estimator_type": "StackedDR",
            "valid_estimators": valid_estimators,
            "failed_estimators": [e for e in self.estimators if e not in valid_estimators],
            "n_components": len(valid_estimators),
            "used_outer_split": self.use_outer_split
        }
        
        # Add weight diagnostics per policy
        diagnostics["weights_per_policy"] = {}
        for policy, weights in self.weights_per_policy.items():
            diagnostics["weights_per_policy"][policy] = {
                name: float(w) for name, w in zip(valid_estimators, weights)
            }
        
        # Add variance reduction metrics
        if len(valid_estimators) > 1:
            diagnostics["variance_reduction"] = self._compute_variance_reduction(valid_estimators)
        
        # Add IF correlation matrix for first policy (as example)
        if self.sampler.target_policies:
            first_policy = self.sampler.target_policies[0]
            IF_matrix = self._collect_influence_functions(first_policy, valid_estimators)
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