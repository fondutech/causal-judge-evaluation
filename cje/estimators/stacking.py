"""Clean implementation of stacked DR estimator using oracle IC approach.

This is a simplified version that removes ~1200 lines of unnecessary complexity:
- No honest CV (weight learning is O_p(n^{-1}))
- No outer splits
- No two-way clustering
- Just the theoretically justified oracle IC approach
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

from cje.estimators.base_estimator import BaseCJEEstimator
from cje.data.models import EstimationResult
from cje.data.precomputed_sampler import PrecomputedSampler

logger = logging.getLogger(__name__)


class StackedDREstimator(BaseCJEEstimator):
    """
    Stacks DR estimators via influence function variance minimization.

    Uses the oracle IC approach: directly computes w^T φ(Z) where w are
    optimal weights. This is theoretically justified because weight learning
    is O_p(n^{-1}) and doesn't affect the asymptotic distribution.

    Key features:
    - Runs multiple DR estimators with shared resources
    - Computes optimal weights via regularized covariance
    - Simple, clean SE computation
    - No unnecessary CV machinery
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        estimators: Optional[List[str]] = None,
        covariance_regularization: float = 1e-4,
        min_weight: float = 0.0,
        weight_shrinkage: float = 0.05,  # Small default shrinkage for stability
        parallel: bool = True,
        seed: int = 42,
        n_folds: int = 5,
        use_iic: bool = False,
        oua_jackknife: bool = True,
        force_uniform_weights: bool = False,  # Hidden flag for sanity checks
        **kwargs: Any,
    ):
        """Initialize the stacked estimator.

        Args:
            sampler: PrecomputedSampler with calibrated data
            estimators: List of estimator names to stack
                Default: ["dr-cpo", "tmle", "mrdr"]
            covariance_regularization: Ridge regularization for numerical stability
            min_weight: Minimum weight for any estimator (for stability)
            weight_shrinkage: Shrinkage toward uniform weights (0=none, 1=uniform)
            parallel: If True, run component estimators in parallel
            seed: Random seed for reproducibility
            n_folds: Number of folds for component cross-fitting
            use_iic: If True, component estimators use IIC
            oua_jackknife: If True, enable OUA for component estimators
            force_uniform_weights: If True, force uniform weights (for debugging)
            **kwargs: Additional arguments (reward_calibrator, etc.)
        """
        super().__init__(sampler)

        # Configuration
        self.estimators = estimators or [
            "dr-cpo",
            "tmle",
            "mrdr",
            "oc-dr-cpo",
            "tr-cpo-e",
        ]
        self.covariance_regularization = covariance_regularization
        self.min_weight = min_weight
        self.weight_shrinkage = weight_shrinkage
        self.parallel = parallel
        self.seed = seed
        self.n_folds = n_folds
        self.use_iic = use_iic
        self.oua_jackknife = oua_jackknife
        self.force_uniform_weights = force_uniform_weights

        # Extract additional configuration
        self.reward_calibrator = kwargs.pop("reward_calibrator", None)
        self.use_calibrated_weights = kwargs.pop("use_calibrated_weights", True)
        self.weight_mode = kwargs.pop("weight_mode", "hajek")

        # Storage for results
        self.component_results: Dict[str, Optional[EstimationResult]] = {}
        self.component_estimators: Dict[str, Any] = {}
        self.weights_per_policy: Dict[str, np.ndarray] = {}
        self.diagnostics_per_policy: Dict[str, Dict] = {}
        self._influence_functions: Dict[str, np.ndarray] = {}

        # Set random seed
        np.random.seed(self.seed)

    def fit(self) -> None:
        """Fit is a no-op for stacked estimator (components fit themselves)."""
        self._fitted = True

    def add_fresh_draws(self, policy: str, fresh_draws_df: Any) -> None:
        """Add fresh draws for DR components.

        This is passed through to component estimators.

        Args:
            policy: The policy name
            fresh_draws_df: Fresh draws data
        """
        # Store fresh draws to pass to components
        if not hasattr(self, "_fresh_draws_per_policy"):
            self._fresh_draws_per_policy = {}
        self._fresh_draws_per_policy[policy] = fresh_draws_df

    def estimate(self) -> EstimationResult:
        """Run all component estimators and stack them optimally."""
        if not self._fitted:
            self.fit()

        # Step 1: Run all component estimators
        self._run_component_estimators()

        # Check for failures
        valid_estimators = [
            name
            for name, result in self.component_results.items()
            if result is not None
        ]

        if len(valid_estimators) == 0:
            raise RuntimeError("All component estimators failed")

        if len(valid_estimators) == 1:
            logger.warning(
                f"Only one valid estimator ({valid_estimators[0]}), using it directly"
            )
            result = self.component_results[valid_estimators[0]]
            if result is None:
                raise RuntimeError("Single valid estimator returned None")
            return result

        # Step 2: Stack estimates for each policy
        stacked_estimates = []
        stacked_ses = []
        stacked_ifs = {}

        for policy_idx, policy in enumerate(self.sampler.target_policies):
            stack_result = self._stack_policy(policy, policy_idx, valid_estimators)

            if stack_result is None:
                stacked_estimates.append(np.nan)
                stacked_ses.append(np.nan)
            else:
                estimate, se, stacked_if = stack_result
                stacked_estimates.append(estimate)
                stacked_ses.append(se)
                stacked_ifs[policy] = stacked_if
                self._influence_functions[policy] = stacked_if

        # Step 3: Create result with diagnostics
        metadata = self._build_metadata(valid_estimators)

        # Get n_samples_used from first valid component
        n_samples_used = {}
        if valid_estimators:
            first_result = self.component_results[valid_estimators[0]]
            if first_result:
                n_samples_used = first_result.n_samples_used

        result = EstimationResult(
            estimates=np.array(stacked_estimates),
            standard_errors=np.array(stacked_ses),
            n_samples_used=n_samples_used,
            method=f"StackedDR({', '.join(valid_estimators)})",
            influence_functions=stacked_ifs,
            metadata=metadata,
            diagnostics=None,
            robust_standard_errors=None,
            robust_confidence_intervals=None,
        )

        # Add any diagnostics from components
        result.diagnostics = self._build_diagnostics(valid_estimators, result)

        # Apply stacked-level oracle jackknife by linear combination of component jackknifes
        self._apply_stacked_oua(result)

        return result

    def _apply_stacked_oua(self, result: EstimationResult) -> None:
        """Augment stacked SEs with oracle jackknife (OUA) via linear combination.

        For fixed stacking weights w, the stacked jackknife path is ψ_stack^(−f) = Σ w_k ψ_k^(−f).
        We compute var_oracle_stack = (K-1)/K * Var_f(ψ_stack^(−f)) and set
        robust_standard_errors = sqrt(standard_errors^2 + var_oracle_stack).
        If no component provides jackknife or K<2, we leave robust_standard_errors as None.
        """
        try:
            policies = list(self.sampler.target_policies)
        except Exception:
            policies = (
                list(result.influence_functions.keys())
                if result.influence_functions
                else []
            )

        if not policies:
            return

        robust_ses: List[float] = []
        var_oracle_per_policy: Dict[str, float] = {}
        jackknife_counts: Dict[str, int] = {}
        contributors_map: Dict[str, List[str]] = {}

        for idx, policy in enumerate(policies):
            # Retrieve weights and component names used for this policy
            weights = self.weights_per_policy.get(policy)
            diag = self.diagnostics_per_policy.get(policy, {})
            used_names = diag.get("components", []) if isinstance(diag, dict) else []

            base_se = (
                float(result.standard_errors[idx])
                if idx < len(result.standard_errors)
                else float("nan")
            )
            var_oracle = 0.0
            K = 0
            contributors: List[str] = []

            if weights is not None and used_names:
                # Collect component jackknife arrays
                jack_list: List[Tuple[float, np.ndarray]] = []
                for w, name in zip(weights, used_names):
                    comp = self.component_estimators.get(name)
                    if not comp or not hasattr(comp, "get_oracle_jackknife"):
                        continue
                    try:
                        jarr = comp.get_oracle_jackknife(policy)
                    except Exception:
                        jarr = None
                    if jarr is None or len(jarr) == 0:
                        continue
                    jack_list.append((float(w), np.asarray(jarr, dtype=float)))
                    contributors.append(name)

                if jack_list:
                    # Align lengths by truncating to minimum K
                    K = min(arr.shape[0] for _, arr in jack_list)
                    if K >= 2:
                        stacked_jack = np.zeros(K, dtype=float)
                        for w, arr in jack_list:
                            stacked_jack += w * arr[:K]
                        mu = float(np.mean(stacked_jack))
                        var_oracle = ((K - 1) / K) * float(
                            np.mean((stacked_jack - mu) ** 2)
                        )

            # Combine with base SE if we have a valid var_oracle
            if np.isfinite(base_se) and var_oracle >= 0.0:
                robust_ses.append(float(np.sqrt(base_se**2 + var_oracle)))
            else:
                robust_ses.append(base_se)

            var_oracle_per_policy[policy] = var_oracle
            jackknife_counts[policy] = int(K)
            contributors_map[policy] = contributors

        # If we computed anything meaningful, attach to result
        if any(k >= 2 for k in jackknife_counts.values()):
            result.robust_standard_errors = np.asarray(robust_ses, dtype=float)
            # Merge into metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata.setdefault("oua", {})
            result.metadata["oua"].update(
                {
                    "source": "stacked-linear-combo",
                    "var_oracle_per_policy": var_oracle_per_policy,
                    "jackknife_counts": jackknife_counts,
                    "contributors": contributors_map,
                }
            )

    def _stack_policy(
        self, policy: str, policy_idx: int, valid_estimators: List[str]
    ) -> Optional[Tuple[float, float, np.ndarray]]:
        """Stack estimates for a single policy.

        Returns:
            Tuple of (estimate, se, stacked_if) or None if failed
        """
        # Collect and align influence functions
        IF_matrix, used_names = self._collect_aligned_ifs(policy, valid_estimators)

        if IF_matrix is None:
            logger.warning(f"No valid influence functions for policy {policy}")
            return None

        if IF_matrix.shape[1] == 0:
            logger.warning(f"No valid influence functions for policy {policy}")
            return None

        # Clean IF matrix: remove rows with NaN/Inf
        # Cast for mypy - we've already checked IF_matrix is not None above
        IF_matrix_clean = cast(np.ndarray, IF_matrix)

        valid_rows = np.all(np.isfinite(IF_matrix_clean), axis=1)
        if not np.all(valid_rows):
            n_dropped = (~valid_rows).sum()
            logger.debug(f"Dropping {n_dropped} rows with NaN/Inf for {policy}")
            IF_matrix_clean = IF_matrix_clean[valid_rows]

        if IF_matrix_clean.shape[0] < 10:
            logger.warning(
                f"Too few valid rows ({IF_matrix_clean.shape[0]}) for {policy}"
            )
            return None

        # Harmonize scales and signs if needed
        IF_matrix_clean = self._harmonize_if_scales(IF_matrix_clean, used_names)
        IF_matrix_clean = self._align_if_signs(IF_matrix_clean)

        # Compute optimal weights with regularization
        weights, weight_diagnostics = self._compute_optimal_weights(IF_matrix_clean)
        self.weights_per_policy[policy] = weights
        self.diagnostics_per_policy[policy] = weight_diagnostics

        # Oracle IC approach: simple and theoretically justified
        stacked_if = IF_matrix_clean @ weights
        se_if = np.std(stacked_if, ddof=1) / np.sqrt(len(stacked_if))

        # Add MC variance if present
        mc_var = self._aggregate_mc_variance(policy, used_names, weights)
        se = float(np.sqrt(se_if**2 + mc_var))

        # Compute point estimate
        component_estimates = []
        for est_name in used_names:
            result = self.component_results[est_name]
            if result:
                component_estimates.append(result.estimates[policy_idx])

        if component_estimates:
            estimate = np.dot(weights, component_estimates)
        else:
            estimate = np.nan

        # Store diagnostics
        weight_diagnostics["se_if"] = se_if
        weight_diagnostics["mc_var"] = mc_var
        weight_diagnostics["n_samples"] = len(stacked_if)
        weight_diagnostics["components"] = used_names

        return estimate, se, stacked_if

    def _compute_optimal_weights(
        self, IF_matrix: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal stacking weights with regularization.

        Solves: min_α α^T Σ α  s.t.  α ≥ 0, 1^T α = 1

        Returns:
            weights: K-dimensional weight vector
            diagnostics: Dictionary with condition numbers, eigenvalues, etc.
        """
        K = IF_matrix.shape[1]

        # Check if forcing uniform weights (for debugging/sanity checks)
        if self.force_uniform_weights:
            w = np.ones(K) / K
            diagnostics = {
                "forced_uniform": True,
                "weights": w.tolist(),
            }
            return w, diagnostics

        # Compute covariance matrix
        centered_IF = IF_matrix - IF_matrix.mean(axis=0, keepdims=True)
        Sigma = np.cov(centered_IF.T)

        # Check condition number before regularization
        eigenvalues = np.linalg.eigvalsh(Sigma)
        condition_pre = eigenvalues.max() / max(eigenvalues.min(), 1e-10)

        # Add regularization for numerical stability
        Sigma_reg = Sigma + self.covariance_regularization * np.eye(K)

        # Check condition number after regularization
        eigenvalues_post = np.linalg.eigvalsh(Sigma_reg)
        condition_post = eigenvalues_post.max() / eigenvalues_post.min()

        # Warn if matrix is ill-conditioned
        if condition_pre > 1e6:
            logger.warning(
                f"Covariance matrix is ill-conditioned (κ={condition_pre:.2e} → "
                f"{condition_post:.2e} after regularization)"
            )

        # Solve for optimal weights
        ones = np.ones(K)
        try:
            w = np.linalg.solve(Sigma_reg, ones)
            w = w / w.sum()

            # Apply minimum weight constraint if specified
            if self.min_weight > 0:
                w = np.maximum(w, self.min_weight)
                w = w / w.sum()

            # Apply shrinkage toward uniform if specified
            if self.weight_shrinkage > 0:
                u = ones / K
                gamma = np.clip(self.weight_shrinkage, 0.0, 1.0)
                w = (1 - gamma) * w + gamma * u
                w = w / w.sum()

        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using uniform weights")
            w = ones / K

        # Compute pairwise correlations between estimators
        correlations = np.corrcoef(centered_IF.T)
        max_corr = np.max(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))

        diagnostics = {
            "condition_pre": condition_pre,
            "condition_post": condition_post,
            "eigenvalues": eigenvalues.tolist(),
            "eigenvalues_post": eigenvalues_post.tolist(),
            "min_eigenvalue": float(eigenvalues.min()),
            "max_eigenvalue": float(eigenvalues.max()),
            "regularization_used": self.covariance_regularization,
            "max_pairwise_correlation": float(max_corr),
            "weights": w.tolist(),
            "weight_shrinkage": self.weight_shrinkage,
            "min_weight": float(w.min()),
            "max_weight": float(w.max()),
        }

        return w, diagnostics

    def _run_component_estimators(self) -> None:
        """Run all component estimators in parallel or sequentially."""
        if self.parallel:
            self._run_parallel()
        else:
            self._run_sequential()

    def _run_parallel(self) -> None:
        """Run component estimators in parallel."""
        with ThreadPoolExecutor(max_workers=len(self.estimators)) as executor:
            futures = {}
            for est_name in self.estimators:
                future = executor.submit(self._run_single_estimator, est_name)
                futures[future] = est_name

            for future in as_completed(futures):
                est_name = futures[future]
                try:
                    result, estimator = future.result()
                    self.component_results[est_name] = result
                    self.component_estimators[est_name] = estimator
                    if result:
                        logger.info(f"Successfully ran {est_name}")
                except Exception as e:
                    logger.error(f"Failed to run {est_name}: {e}")
                    self.component_results[est_name] = None

    def _run_sequential(self) -> None:
        """Run component estimators sequentially."""
        for est_name in self.estimators:
            try:
                result, estimator = self._run_single_estimator(est_name)
                self.component_results[est_name] = result
                self.component_estimators[est_name] = estimator
                if result:
                    logger.info(f"Successfully ran {est_name}")
            except Exception as e:
                logger.error(f"Failed to run {est_name}: {e}")
                self.component_results[est_name] = None

    def _run_single_estimator(self, est_name: str) -> Tuple[EstimationResult, Any]:
        """Run a single component estimator."""
        # Import estimators here to avoid circular imports
        from cje.estimators.dr_base import DRCPOEstimator
        from cje.estimators.tmle import TMLEEstimator
        from cje.estimators.mrdr import MRDREstimator
        from cje.estimators.orthogonalized_calibrated_dr import (
            OrthogonalizedCalibratedDRCPO,
        )
        from cje.estimators.tr_cpo import TRCPOEstimator

        estimator_map = {
            "dr-cpo": DRCPOEstimator,
            "tmle": TMLEEstimator,
            "mrdr": MRDREstimator,
            "oc-dr-cpo": OrthogonalizedCalibratedDRCPO,
            "tr-cpo-e": TRCPOEstimator,  # efficient TR-CPO (m̂(S))
        }

        if est_name not in estimator_map:
            raise ValueError(f"Unknown estimator: {est_name}")

        EstimatorClass = estimator_map[est_name]

        # Create estimator with shared configuration
        # TR-CPO has different constructor
        if est_name == "tr-cpo-e":
            estimator = EstimatorClass(
                sampler=self.sampler,
                reward_calibrator=self.reward_calibrator,
                n_folds=self.n_folds,
                use_iic=self.use_iic,
                oua_jackknife=self.oua_jackknife,
            )
        else:
            estimator = EstimatorClass(
                sampler=self.sampler,
                reward_calibrator=self.reward_calibrator,
                n_folds=self.n_folds,
                use_calibrated_weights=self.use_calibrated_weights,
                weight_mode=self.weight_mode,
                use_iic=self.use_iic,
                oua_jackknife=self.oua_jackknife,
            )

        # Pass fresh draws if available
        if hasattr(self, "_fresh_draws_per_policy"):
            for policy, fresh_draws in self._fresh_draws_per_policy.items():
                estimator.add_fresh_draws(policy, fresh_draws)

        # Run estimation
        result = estimator.fit_and_estimate()
        return result, estimator

    def _collect_aligned_ifs(
        self, policy: str, valid_estimators: List[str]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """Collect and align influence functions from components.

        Returns:
            IF_matrix: n x K matrix of aligned IFs
            used_names: List of estimator names that contributed IFs
        """
        # Find common sample indices across all components
        common_indices = None
        for est_name in valid_estimators:
            result = self.component_results[est_name]
            if (
                result
                and result.influence_functions
                and policy in result.influence_functions
            ):
                if_data = result.influence_functions[policy]
                if if_data is not None and len(if_data) > 0:
                    # Get indices from metadata if available
                    indices = None
                    if hasattr(result, "metadata") and isinstance(
                        result.metadata, dict
                    ):
                        if_indices = result.metadata.get("if_sample_indices", {})
                        indices = if_indices.get(policy)

                    if indices is not None:
                        if common_indices is None:
                            common_indices = set(indices)
                        else:
                            common_indices = common_indices.intersection(indices)

        if common_indices is None or len(common_indices) == 0:
            logger.warning(f"No common samples found for {policy}")
            return None, []

        # Collect aligned IFs
        IF_columns = []
        used_names = []

        for est_name in valid_estimators:
            result = self.component_results[est_name]
            if (
                result
                and result.influence_functions
                and policy in result.influence_functions
            ):
                if_data = result.influence_functions[policy]
                if if_data is not None and len(if_data) > 0:
                    # Get indices for alignment
                    if hasattr(result, "metadata") and isinstance(
                        result.metadata, dict
                    ):
                        if_indices = result.metadata.get("if_sample_indices", {})
                        indices = if_indices.get(policy)

                        if indices is not None:
                            # Create alignment mapping
                            idx_to_pos = {idx: i for i, idx in enumerate(indices)}
                            aligned_if = np.zeros(len(common_indices))

                            for j, common_idx in enumerate(sorted(common_indices)):
                                if common_idx in idx_to_pos:
                                    aligned_if[j] = if_data[idx_to_pos[common_idx]]

                            IF_columns.append(aligned_if)
                            used_names.append(est_name)

        if len(IF_columns) == 0:
            return None, []

        IF_matrix = np.column_stack(IF_columns)
        return IF_matrix, used_names

    def _harmonize_if_scales(
        self, IF_matrix: np.ndarray, used_names: List[str]
    ) -> np.ndarray:
        """Ensure all IFs are on the same scale."""
        # Simple check: if SEs differ by more than 50%, there might be a scale issue
        for i in range(IF_matrix.shape[1]):
            col_se = np.std(IF_matrix[:, i], ddof=1) / np.sqrt(len(IF_matrix[:, i]))
            expected_se = col_se  # This would come from component metadata

            # For now, we trust that components are providing correctly scaled IFs
            # Could add more sophisticated checks here if needed

        return IF_matrix

    def _align_if_signs(self, IF_matrix: np.ndarray) -> np.ndarray:
        """Align signs of IFs to ensure they're measuring the same direction."""
        if IF_matrix.shape[1] <= 1:
            return IF_matrix

        # Use first column as reference
        reference = IF_matrix[:, 0]

        for i in range(1, IF_matrix.shape[1]):
            correlation = np.corrcoef(reference, IF_matrix[:, i])[0, 1]
            if correlation < 0:
                logger.debug(
                    f"Flipping sign of component {i} (correlation={correlation:.3f})"
                )
                IF_matrix[:, i] = -IF_matrix[:, i]

        return IF_matrix

    def _aggregate_mc_variance(
        self, policy: str, used_names: List[str], weights: np.ndarray
    ) -> float:
        """Aggregate Monte Carlo variance from components."""
        mc_vars = []

        for est_name in used_names:
            result = self.component_results[est_name]
            if (
                result
                and hasattr(result, "metadata")
                and isinstance(result.metadata, dict)
            ):
                mc_var = result.metadata.get("mc_variance", {}).get(policy, 0.0)
                mc_vars.append(mc_var)
            else:
                mc_vars.append(0.0)

        if mc_vars:
            # Weighted sum of MC variances
            return float(np.dot(weights**2, mc_vars))
        return 0.0

    def _build_metadata(self, valid_estimators: List[str]) -> Dict[str, Any]:
        """Build metadata for the result."""
        metadata = {
            "stacking_weights": self.weights_per_policy,
            "stacking_diagnostics": self.diagnostics_per_policy,
            "valid_estimators": valid_estimators,
            "failed_estimators": [
                e for e in self.estimators if e not in valid_estimators
            ],
            "component_iic": self.use_iic,
            "n_folds": self.n_folds,
            "covariance_regularization": self.covariance_regularization,
        }

        # Add sample indices if available
        if_sample_indices: Dict[str, Any] = {}
        for name in valid_estimators:
            result = self.component_results[name]
            if (
                result
                and hasattr(result, "metadata")
                and isinstance(result.metadata, dict)
            ):
                comp_indices = result.metadata.get("if_sample_indices", {})
                if comp_indices and not if_sample_indices:
                    if_sample_indices = comp_indices
                    break

        if if_sample_indices:
            metadata["if_sample_indices"] = if_sample_indices

        return metadata

    def _build_diagnostics(
        self, valid_estimators: List[str], result: EstimationResult
    ) -> Optional[Any]:
        """Build diagnostics object if applicable."""
        # Could aggregate diagnostics from components if needed
        # For now, return None as stacking has its own diagnostics in metadata
        return None
