"""
IPS-only estimators for CJE.

This module contains all inverse propensity scoring (IPS) estimators that
do not use outcome modeling. These are simpler but can have higher variance
than doubly-robust methods.

Available estimators:
- IPS: Standard inverse propensity scoring
- SNIPS: Self-normalized IPS (reduces variance, adds small bias)
- CalibratedIPS: IPS with isotonic calibration of weights
"""

from typing import Dict, List, Any, Optional
import numpy as np
from rich.console import Console
from sklearn.model_selection import KFold

from ..loggers import MultiTargetSampler
from ..calibration import calibrate_weights_isotonic
from .results import EstimationResult
from .reliability import EstimatorMetadata
from .base import Estimator

console = Console()


class IPS(Estimator[Dict[str, Any]]):
    """
    Standard Inverse Propensity Scoring estimator.

    Uses raw importance weights without any outcome modeling.
    This is the simplest off-policy estimator but can have high variance.

    Args:
        sampler: Multi-target sampler for computing log probabilities
        stabilize_weights: Whether to apply numerical stabilization for extreme log differences (default: True)
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        stabilize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.stabilize_weights = stabilize_weights
        self.n: int = 0
        self.K: int = sampler.K
        self._weights_matrix: Optional[np.ndarray] = None  # Shape (n, K)
        self._rewards: Optional[np.ndarray] = None  # Shape (n,)
        self._weight_stats: Optional[Dict[str, Any]] = None  # Weight statistics

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """
        Fit IPS estimator to logged data.

        Args:
            logs: List of logged data points with required fields:
                  - context: Input context
                  - response: Generated sequence
                  - logp: Log probability under behavior policy
                  - reward: Observed reward
        """
        self.n = len(logs)

        # Extract data
        contexts = [log["context"] for log in logs]
        responses = [log["response"] for log in logs]
        self._rewards = np.array([log["reward"] for log in logs])
        logp_behavior = np.array([log["logp"] for log in logs])

        # Debug: Check log probability ranges
        console.print(
            f"[blue]Behavior log probs range: [{np.min(logp_behavior):.2f}, {np.max(logp_behavior):.2f}][/blue]"
        )

        # Compute importance weights
        console.print(
            f"[bold blue]Computing importance weights for {self.K} policies...[/bold blue]"
        )

        (
            self._weights_matrix,
            self._weight_stats,
        ) = self.sampler.importance_weights_matrix(
            contexts,
            responses,
            logp_behavior.tolist(),
            stabilize=self.stabilize_weights,
            return_stats=True,
        )

    def estimate(self) -> EstimationResult:
        """
        Return multi-policy IPS estimates.

        Returns:
            EstimationResult containing:
                - v_hat: Policy value estimates for each policy
                - se: Standard errors via bootstrap
                - eif_components: Efficient influence function values
                - metadata: Estimation metadata including ESS
        """
        # Normalize weights
        W_normalized = self._weights_matrix / np.mean(self._weights_matrix, axis=0)

        # IPS estimate: mean of weighted rewards
        v_hat = np.mean(
            W_normalized * self._rewards[:, np.newaxis], axis=0
        )  # Shape: (K,)

        # EIF components for IPS: w * r - v_hat
        eif_components = W_normalized * self._rewards[:, np.newaxis] - v_hat

        # Bootstrap standard errors
        n_bootstrap = 1000
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(self.n, self.n, replace=True)
            boot_estimate = np.mean(eif_components[idx], axis=0)
            bootstrap_estimates.append(boot_estimate)

        se = np.std(bootstrap_estimates, axis=0)

        # Covariance matrix
        if self.K == 1:
            # For single policy, compute variance directly
            Sigma_hat = np.array([[np.var(eif_components[:, 0], ddof=1) / self.n]])
        else:
            Sigma_hat = np.cov(eif_components.T) / self.n

        # Create metadata
        metadata = EstimatorMetadata(
            estimator_type="IPS",
            stabilize_weights=self.stabilize_weights,
            bootstrap_available=True,
        )

        # Add weight statistics to metadata
        if self._weight_stats:
            metadata.weight_range = self._weight_stats.get("weight_range")
            metadata.ess_values = self._weight_stats.get("ess_values")
            metadata.ess_percentage = self._weight_stats.get("ess_percentage")
            metadata.n_clipped = self._weight_stats.get("n_clipped", 0)
            metadata.clip_fraction = self._weight_stats.get("clip_fraction", 0.0)
            metadata.stabilization_applied = self._weight_stats.get(
                "stabilization_applied", False
            )

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            n_policies=self.K,
            eif_components=eif_components,
            covariance_matrix=Sigma_hat,
            metadata=metadata.to_dict(),
        )


class SNIPS(IPS):
    """
    Self-Normalized Inverse Propensity Scoring estimator.

    SNIPS normalizes the importance weights to sum to 1, which can reduce
    variance at the cost of introducing a small bias. This is particularly
    useful when the behavior and target policies are very different.

    The SNIPS estimate is:
    v_hat(π_k) = Σ(w_i * r_i) / Σ(w_i)

    where w_i are the importance weights for policy k.

    Args:
        sampler: Multi-target sampler for computing log probabilities
        stabilize_weights: Whether to apply numerical stabilization (default: True)
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        stabilize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(sampler, stabilize_weights=stabilize_weights, **kwargs)

    def estimate(self) -> EstimationResult:
        """
        Return self-normalized IPS estimates.

        Returns:
            EstimationResult with self-normalized estimates
        """
        if self._weights_matrix is None or self._rewards is None:
            raise RuntimeError("Must call fit() before estimate()")

        # SNIPS estimate: normalize weights to sum to 1
        v_hat = np.zeros(self.K)
        eif_components = np.zeros((self.n, self.K))

        for k in range(self.K):
            weights_k = self._weights_matrix[:, k]

            # Self-normalization: weights sum to 1
            weights_normalized = weights_k / np.sum(weights_k)

            # SNIPS estimate
            v_hat[k] = np.sum(weights_normalized * self._rewards)

            # EIF for SNIPS (approximation - exact EIF is more complex)
            # This uses the delta method approximation
            mean_weight = np.mean(weights_k)
            eif_components[:, k] = (
                weights_k * self._rewards / mean_weight
                - weights_k * v_hat[k] / mean_weight
            ) / self.n

        # Bootstrap standard errors
        n_bootstrap = 1000
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(self.n, self.n, replace=True)
            boot_v = np.zeros(self.K)
            for k in range(self.K):
                weights_k_boot = self._weights_matrix[idx, k]
                weights_norm_boot = weights_k_boot / np.sum(weights_k_boot)
                boot_v[k] = np.sum(weights_norm_boot * self._rewards[idx])
            bootstrap_estimates.append(boot_v)

        se = np.std(bootstrap_estimates, axis=0)

        # Covariance matrix
        if self.K == 1:
            # For single policy, compute variance directly
            Sigma_hat = np.array([[np.var(eif_components[:, 0], ddof=1) / self.n]])
        else:
            Sigma_hat = np.cov(eif_components.T) / self.n

        # Create metadata
        metadata = EstimatorMetadata(
            estimator_type="SNIPS",
            stabilize_weights=self.stabilize_weights,
            bootstrap_available=True,
        )

        # Add weight statistics
        if self._weight_stats:
            metadata.weight_range = self._weight_stats.get("weight_range")
            metadata.ess_values = self._weight_stats.get("ess_values")
            metadata.ess_percentage = self._weight_stats.get("ess_percentage")
            metadata.n_clipped = self._weight_stats.get("n_clipped", 0)
            metadata.clip_fraction = self._weight_stats.get("clip_fraction", 0.0)

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            n_policies=self.K,
            eif_components=eif_components,
            covariance_matrix=Sigma_hat,
            metadata=metadata.to_dict(),
        )


class CalibratedIPS(IPS):
    """
    Calibrated IPS estimator using CJE's isotonic calibration.

    This estimator improves on standard IPS by applying the library's
    weight calibration pipeline which includes:
    1. Clipping of extreme weights (done first)
    2. Isotonic calibration to achieve target mean = 1
    3. Cross-fitting to avoid overfitting

    This is the recommended default estimator when no target policy samples
    are available for outcome modeling, as it reduces variance while
    maintaining approximately unbiased estimates.

    Args:
        sampler: Multi-target sampler for computing log probabilities
        clip_min: Minimum weight after clipping (default: 0.01)
        clip_max: Maximum weight after clipping (default: 100.0)
        n_folds: Number of folds for cross-fitting calibration (default: 5)
        max_calibrated_weight: Hard cap after calibration (default: 500.0)
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        clip_min: float = 0.01,
        clip_max: float = 100.0,
        n_folds: int = 5,
        max_calibrated_weight: float = 500.0,
        **kwargs: Any,
    ) -> None:
        # Don't use built-in stabilization, we'll handle it via calibration
        super().__init__(sampler, stabilize_weights=False, **kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.n_folds = n_folds
        self.max_calibrated_weight = max_calibrated_weight
        self._calibration_diagnostics: Optional[Dict[str, Any]] = None

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """
        Fit calibrated IPS estimator to logged data.

        Args:
            logs: List of logged data points
        """
        self.n = len(logs)

        # Extract data
        contexts = [log["context"] for log in logs]
        responses = [log["response"] for log in logs]
        self._rewards = np.array([log["reward"] for log in logs])
        logp_behavior = np.array([log["logp"] for log in logs])

        console.print(
            f"[bold blue]Computing calibrated importance weights for {self.K} policies...[/bold blue]"
        )

        # Compute raw importance weights
        raw_weights, weight_stats = self.sampler.importance_weights_matrix(
            contexts,
            responses,
            logp_behavior.tolist(),
            stabilize=False,  # We'll apply our own calibration
            return_stats=True,
        )

        # Apply clipping BEFORE calibration to prevent extreme weights from distorting calibration
        clipped_weights = np.clip(raw_weights, self.clip_min, self.clip_max)

        # Create fold indices for cross-fitting (if n > 1)
        if self.n_folds > 1:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            fold_indices = np.zeros(self.n, dtype=int)
            for fold_idx, (_, val_idx) in enumerate(kf.split(range(self.n))):
                fold_indices[val_idx] = fold_idx
        else:
            fold_indices = None

        # Apply isotonic calibration using library function
        # This ensures weights have mean=1 while maintaining monotonicity
        self._weights_matrix, self._calibration_diagnostics = (
            calibrate_weights_isotonic(
                clipped_weights,
                fold_indices=fold_indices,
                target_mean=1.0,
                max_calibrated_weight=self.max_calibrated_weight,
                min_samples_for_calibration=10,
            )
        )

        # Log calibration results
        console.print(
            f"[green]Calibration complete. "
            f"Mean weights achieved: {self._calibration_diagnostics.get('mean_achieved', [])}"
            f"[/green]"
        )

        if self._calibration_diagnostics.get("warnings"):
            for warning in self._calibration_diagnostics["warnings"]:
                console.print(f"[yellow]Warning: {warning}[/yellow]")

        # Store weight statistics
        self._weight_stats = {
            "weight_range": [
                (
                    float(np.min(self._weights_matrix[:, k])),
                    float(np.max(self._weights_matrix[:, k])),
                )
                for k in range(self.K)
            ],
            "n_clipped_low": int(np.sum(raw_weights < self.clip_min)),
            "n_clipped_high": int(np.sum(raw_weights > self.clip_max)),
            "clip_fraction": float(
                np.mean((raw_weights < self.clip_min) | (raw_weights > self.clip_max))
            ),
        }

        # Compute ESS for each policy
        ess_values = []
        for k in range(self.K):
            weights_k = self._weights_matrix[:, k]
            if np.sum(weights_k) > 0:
                ess = np.sum(weights_k) ** 2 / np.sum(weights_k**2)
            else:
                ess = 0.0
            ess_values.append(float(ess))

        self._weight_stats["ess_values"] = ess_values
        self._weight_stats["ess_percentage"] = [
            100.0 * ess / self.n for ess in ess_values
        ]

        console.print(
            f"[green]Average ESS: {np.mean(ess_values):.1f} "
            f"({np.mean(self._weight_stats['ess_percentage']):.1f}%)[/green]"
        )

    def estimate(self) -> EstimationResult:
        """
        Return calibrated IPS estimates.

        Returns:
            EstimationResult with calibrated and clipped estimates
        """
        # Get base estimate using parent class
        result = super().estimate()

        # Update metadata to reflect this is CalibratedIPS
        metadata = EstimatorMetadata(
            estimator_type="CalibratedIPS",
            clip_threshold=self.clip_max,
            k_folds=self.n_folds,
            bootstrap_available=True,
            calibrate_weights=True,
        )

        # Add weight statistics
        if self._weight_stats:
            metadata.weight_range = self._weight_stats.get("weight_range")
            metadata.ess_values = self._weight_stats.get("ess_values")
            metadata.ess_percentage = self._weight_stats.get("ess_percentage")
            metadata.n_clipped = self._weight_stats.get(
                "n_clipped_low", 0
            ) + self._weight_stats.get("n_clipped_high", 0)
            metadata.clip_fraction = self._weight_stats.get("clip_fraction", 0.0)

        result.metadata = metadata.to_dict()

        # Add calibration diagnostics
        if self._calibration_diagnostics:
            result.metadata["calibration_diagnostics"] = self._calibration_diagnostics
        result.metadata["clip_min"] = self.clip_min
        result.metadata["clip_max"] = self.clip_max
        result.metadata["max_calibrated_weight"] = self.max_calibrated_weight

        return result


# Aliases for backward compatibility and convenience
MultiIPSEstimator = IPS
MultiSNIPSEstimator = SNIPS
CalibratedIPSEstimator = CalibratedIPS
