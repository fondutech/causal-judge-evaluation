"""
Inverse Propensity Scoring (IPS) and Self-Normalized IPS (SNIPS) estimators.

This module contains multi-policy estimators for the CJE framework's unified architecture.
All estimators handle multiple policies, with single-policy evaluation being treated as the K=1 case.

**Architecture:**
- MultiIPSEstimator: Native multi-policy implementation
- MultiSNIPSEstimator: Native multi-policy implementation
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional
from .base import Estimator
from .results import EstimationResult
from ..loggers.multi_target_sampler import MultiTargetSampler


class MultiIPSEstimator(Estimator[Dict[str, Any]]):
    """
    Multi-policy IPS estimator for K target policies.

    Estimates values for multiple target policies {π¹, π², ..., πᴷ} simultaneously:
    V_hat_k = (1/n) * Σ w_i_k * r_i
    where w_i_k = π^k(s_i|x_i) / π_0(s_i|x_i) is the importance ratio for policy k.

    Args:
        sampler: MultiTargetSampler instance for K target policies
        clip: Optional maximum value for importance weights (None for no clipping - research mode)
        stabilize_weights: Whether to apply numerical stabilization for extreme log differences (default: True)
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        clip: Optional[float] = None,
        stabilize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sampler = sampler
        self.clip = clip
        self.stabilize_weights = stabilize_weights
        self.n: int = 0
        self.K: int = sampler.K
        self._weights_matrix: np.ndarray | None = None  # Shape (n, K)
        self._rewards: np.ndarray | None = None  # Shape (n,)
        self._weight_stats: Optional[Dict[str, Any]] = None  # Weight statistics

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """Fit the multi-policy IPS estimator.

        Args:
            logs: List of logged data points with required fields:
                  - context: Input context
                  - response: Generated sequence
                  - logp: Log probability under behavior policy
                  - reward: Observed reward
        """
        if not logs:
            raise ValueError("Cannot fit estimator with empty logs")

        self.n = len(logs)

        # Extract data
        self._rewards = np.array([log["reward"] for log in logs], dtype=float)
        contexts = [log["context"] for log in logs]
        responses = [log["response"] for log in logs]
        logp_behavior = np.array([log["logp"] for log in logs], dtype=float)

        # Compute importance weights matrix (n, K) with statistics
        self._weights_matrix, self._weight_stats = (
            self.sampler.importance_weights_matrix(
                contexts,
                responses,
                logp_behavior.tolist(),
                clip=self.clip,
                stabilize=self.stabilize_weights,
                return_stats=True,
            )
        )

    def estimate(self) -> EstimationResult:
        """
        Return multi-policy IPS estimates.

        Returns:
            EstimationResult containing:
                v_hat: Point estimates (K,)
                se: Standard errors (K,)
                cov: Covariance matrix (K, K)
                n: Number of samples (scalar)
                K: Number of policies (scalar)
                name: Estimator name
        """
        if self._weights_matrix is None or self._rewards is None:
            raise RuntimeError("Must call fit() before estimate()")

        # Compute point estimates: v_hat_k = (1/n) * Σ w_i_k * r_i
        v_hat = np.mean(
            self._weights_matrix * self._rewards[:, np.newaxis], axis=0
        )  # Shape (K,)

        # Compute EIF matrix: phi_i_k = w_i_k * r_i - v_hat_k
        phi_matrix = (
            self._weights_matrix * self._rewards[:, np.newaxis] - v_hat
        )  # Shape (n, K)

        # Compute covariance matrix
        if self.K == 1:
            # For K=1, ensure we get a 2D array
            cov = np.array([[np.var(phi_matrix[:, 0], ddof=1) / self.n]])
        else:
            cov = np.cov(phi_matrix, rowvar=False) / self.n  # Shape (K, K)

        # Compute standard errors
        se = np.sqrt(np.diag(cov))  # Shape (K,)

        # Create structured metadata
        from .reliability import EstimatorMetadata

        structured_metadata = EstimatorMetadata(
            estimator_type="IPS",
            clip_threshold=self.clip,
            stabilize_weights=self.stabilize_weights,
            bootstrap_available=phi_matrix is not None,
        )

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

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            eif_components=phi_matrix,
            covariance_matrix=cov,
            estimator_type="IPS",
            n_policies=self.K,
            metadata=structured_metadata.to_dict(),
        )


class MultiSNIPSEstimator(Estimator[Dict[str, Any]]):
    """
    Multi-policy SNIPS estimator for K target policies.

    Estimates values for multiple target policies using normalized importance sampling:
    V_hat_k = (Σ w_i_k * r_i) / (Σ w_i_k)
    where w_i_k = π^k(s_i|x_i) / π_0(s_i|x_i) is the importance ratio for policy k.

    Args:
        sampler: MultiTargetSampler instance for K target policies
        clip: Optional maximum value for importance weights (None for no clipping - research mode)
        stabilize_weights: Whether to apply numerical stabilization for extreme log differences (default: True)
    """

    def __init__(
        self,
        sampler: MultiTargetSampler,
        clip: Optional[float] = None,
        stabilize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sampler = sampler
        self.clip = clip
        self.stabilize_weights = stabilize_weights
        self.n: int = 0
        self.K: int = sampler.K
        self._weights_matrix: np.ndarray | None = None  # Shape (n, K)
        self._rewards: np.ndarray | None = None  # Shape (n,)
        self._weight_stats: Optional[Dict[str, Any]] = None  # Weight statistics

    def fit(self, logs: List[Dict[str, Any]], **kwargs: Any) -> None:
        """Fit the multi-policy SNIPS estimator.

        Args:
            logs: List of logged data points with required fields:
                  - context: Input context
                  - response: Generated sequence
                  - logp: Log probability under behavior policy
                  - reward: Observed reward
        """
        if not logs:
            raise ValueError("Cannot fit estimator with empty logs")

        self.n = len(logs)

        # Extract data
        self._rewards = np.array([log["reward"] for log in logs], dtype=float)
        contexts = [log["context"] for log in logs]
        responses = [log["response"] for log in logs]
        logp_behavior = np.array([log["logp"] for log in logs], dtype=float)

        # Compute importance weights matrix (n, K) with statistics
        self._weights_matrix, self._weight_stats = (
            self.sampler.importance_weights_matrix(
                contexts,
                responses,
                logp_behavior.tolist(),
                clip=self.clip,
                stabilize=self.stabilize_weights,
                return_stats=True,
            )
        )

    def estimate(self) -> EstimationResult:
        """
        Return multi-policy SNIPS estimates.

        Returns:
            EstimationResult containing:
                v_hat: Point estimates (K,)
                se: Standard errors (K,)
                cov: Covariance matrix (K, K)
                n: Number of samples (scalar)
                K: Number of policies (scalar)
                name: Estimator name
        """
        if self._weights_matrix is None or self._rewards is None:
            raise RuntimeError("Must call fit() before estimate()")

        # Compute normalized estimates: v_hat_k = (Σ w_i_k * r_i) / (Σ w_i_k)
        w_sums = np.sum(self._weights_matrix, axis=0)  # Shape (K,)

        # Check for zero weight sums
        if np.any(w_sums == 0):
            raise RuntimeError("Sum of importance weights is zero for some policies")

        numerator = np.sum(
            self._weights_matrix * self._rewards[:, np.newaxis], axis=0
        )  # Shape (K,)
        v_hat = numerator / w_sums  # Shape (K,)

        # Compute EIF matrix for SNIPS using the correct formula
        # For SNIPS, the EIF is: phi_i_k = (w_i_k * (r_i - v_hat_k)) / (w_sum_k / n)
        # This is equivalent to: phi_i_k = n * w_i_k * (r_i - v_hat_k) / w_sum_k
        #
        # The key insight is that SNIPS variance should scale with n, not be divided by it.
        # The previous bug divided by w_sum/n twice (once explicitly, once in covariance calculation).

        phi_matrix = (
            self.n * self._weights_matrix * (self._rewards[:, np.newaxis] - v_hat)
        ) / w_sums[
            np.newaxis, :
        ]  # Shape (n, K) - Correct SNIPS EIF

        # Compute covariance matrix
        if self.K == 1:
            # For K=1, ensure we get a 2D array
            cov = np.array([[np.var(phi_matrix[:, 0], ddof=1) / self.n]])
        else:
            cov = np.cov(phi_matrix, rowvar=False) / self.n  # Shape (K, K)

        # Compute standard errors
        se = np.sqrt(np.diag(cov))  # Shape (K,)

        # Create structured metadata
        from .reliability import EstimatorMetadata

        structured_metadata = EstimatorMetadata(
            estimator_type="SNIPS",
            clip_threshold=self.clip,
            stabilize_weights=self.stabilize_weights,
            bootstrap_available=phi_matrix is not None,
        )

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

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            eif_components=phi_matrix,
            covariance_matrix=cov,
            estimator_type="SNIPS",
            n_policies=self.K,
            metadata=structured_metadata.to_dict(),
        )
