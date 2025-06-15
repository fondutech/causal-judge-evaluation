"""
IPS (Inverse Propensity Scoring) estimators for CJE.

This module provides simple IPS estimators as baselines for comparison
with more sophisticated doubly-robust methods.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from tqdm.auto import tqdm
from rich.console import Console

from ..loggers import MultiTargetSampler
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
        Sigma_hat = np.cov(eif_components.T) / self.n

        # Create metadata
        metadata = EstimatorMetadata(
            estimator_type="IPS",
            stabilize_weights=self.stabilize_weights,
            bootstrap_available=True,
        )

        # Add weight statistics to metadata
        if self._weight_stats:
            structured_metadata = metadata.to_dict()
            structured_metadata["n_samples"] = self._weight_stats["n_samples"]
            structured_metadata["n_policies"] = self._weight_stats["n_policies"]
            structured_metadata["weight_range"] = self._weight_stats["weight_range"]
            structured_metadata["ess_values"] = self._weight_stats["ess_values"]
            structured_metadata["ess_percentage"] = self._weight_stats["ess_percentage"]
            structured_metadata["weight_stats"] = self._weight_stats
            metadata = EstimatorMetadata(**structured_metadata)

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            n_policies=self.K,
            eif_components=eif_components,
            covariance_matrix=Sigma_hat,
            metadata=metadata.to_dict(),
        )


class WeightedIPS(IPS):
    """
    Weighted IPS estimator (for trajectory data).

    Extends standard IPS to handle trajectory weights, where each
    trajectory has an associated weight (e.g., based on conversation length).

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
        super().__init__(sampler, stabilize_weights=stabilize_weights, **kwargs)
        self._trajectory_weights: Optional[np.ndarray] = None

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """
        Fit weighted IPS estimator to logged data.

        Args:
            logs: List of logged data points with trajectory weights
        """
        super().fit(logs, **kwargs)

        # Extract trajectory weights if available
        if logs and "trajectory_weight" in logs[0]:
            self._trajectory_weights = np.array(
                [log["trajectory_weight"] for log in logs]
            )
        else:
            # Equal weights if not provided
            self._trajectory_weights = np.ones(self.n)

    def estimate(self) -> EstimationResult:
        """
        Return weighted IPS estimates.

        Returns:
            EstimationResult with trajectory-weighted estimates
        """
        # Get base IPS components
        result = super().estimate()

        # Apply trajectory weights
        if self._trajectory_weights is not None:
            # Normalize trajectory weights
            traj_weights_normalized = self._trajectory_weights / np.mean(
                self._trajectory_weights
            )

            # Reweight EIF components
            eif_weighted = (
                result.eif_components * traj_weights_normalized[:, np.newaxis]
            )

            # Recompute estimates with trajectory weights
            v_hat_weighted = np.mean(eif_weighted + result.v_hat, axis=0)

            # Bootstrap with trajectory weights
            n_bootstrap = 1000
            bootstrap_estimates = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(self.n, self.n, replace=True)
                boot_estimate = np.mean(eif_weighted[idx], axis=0)
                bootstrap_estimates.append(boot_estimate)

            se_weighted = np.std(bootstrap_estimates, axis=0)

            # Update result
            result.v_hat = v_hat_weighted
            result.se = se_weighted
            result.eif_components = eif_weighted
            # Update metadata
            if isinstance(result.metadata, dict):
                result.metadata["estimator_type"] = "WeightedIPS"
            else:
                # Create new metadata dict
                from .reliability import EstimatorMetadata

                metadata = EstimatorMetadata(
                    estimator_type="WeightedIPS",
                    stabilize_weights=self.stabilize_weights,
                    bootstrap_available=True,
                )
                result.metadata = metadata.to_dict()

        return result


# Aliases for backward compatibility and naming consistency
MultiIPSEstimator = IPS
MultiSNIPSEstimator = IPS  # SNIPS is just IPS with normalized weights
