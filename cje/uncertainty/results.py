"""Result objects for uncertainty-aware estimation.

This module provides clean result objects that properly handle multi-policy
evaluation with uncertainty quantification.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass, field

from .schemas import UncertaintyAwareEstimate, VarianceDecomposition
from ..estimators.results import EstimationResult


@dataclass
class PolicyResult:
    """Result for a single policy with uncertainty quantification."""

    name: str
    estimate: UncertaintyAwareEstimate
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def value(self) -> float:
        """Point estimate value."""
        return self.estimate.value

    @property
    def se(self) -> float:
        """Standard error."""
        return self.estimate.se

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval."""
        return (self.estimate.ci_lower, self.estimate.ci_upper)

    def summary(self) -> str:
        """Human-readable summary."""
        return f"{self.name}: {self.estimate.summary()}"


@dataclass
class MultiPolicyUncertaintyResult:
    """Results for multiple policies with uncertainty quantification."""

    policies: List[PolicyResult]
    n_samples: int
    estimator_type: str = "UncertaintyAwareDRCPO"
    global_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and setup derived attributes."""
        if not self.policies:
            raise ValueError("At least one policy result required")

    @property
    def n_policies(self) -> int:
        """Number of policies evaluated."""
        return len(self.policies)

    def get_policy(self, name: str) -> Optional[PolicyResult]:
        """Get result for a specific policy by name."""
        for policy in self.policies:
            if policy.name == name:
                return policy
        return None

    def get_estimates(self) -> np.ndarray:
        """Get all point estimates as array."""
        return np.array([p.value for p in self.policies])

    def get_standard_errors(self) -> np.ndarray:
        """Get all standard errors as array."""
        return np.array([p.se for p in self.policies])

    def get_confidence_intervals(self) -> List[Tuple[float, float]]:
        """Get all confidence intervals."""
        return [p.confidence_interval for p in self.policies]

    def rank_policies(self) -> List[str]:
        """Rank policies by point estimate (descending)."""
        sorted_policies = sorted(self.policies, key=lambda p: p.value, reverse=True)
        return [p.name for p in sorted_policies]

    def pairwise_comparison(self, policy1: str, policy2: str) -> Dict[str, Any]:
        """Compare two policies with uncertainty."""
        p1 = self.get_policy(policy1)
        p2 = self.get_policy(policy2)

        if not p1 or not p2:
            raise ValueError(f"Policy not found: {policy1 if not p1 else policy2}")

        # Difference and its standard error (assuming independence)
        diff = p1.value - p2.value
        se_diff = np.sqrt(p1.se**2 + p2.se**2)

        # Z-score and p-value
        z_score = diff / se_diff if se_diff > 0 else 0
        # Two-tailed p-value using standard normal CDF
        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            "policy1": policy1,
            "policy2": policy2,
            "difference": diff,
            "se_difference": se_diff,
            "z_score": z_score,
            "p_value": p_value,
            "significant_at_0.05": bool(p_value < 0.05),
            "favors": policy1 if diff > 0 else policy2,
        }

    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            f"Multi-Policy Uncertainty-Aware Results",
            f"=" * 50,
            f"Estimator: {self.estimator_type}",
            f"Samples: {self.n_samples}",
            f"Policies: {self.n_policies}",
            "",
            "Rankings:",
        ]

        for i, name in enumerate(self.rank_policies(), 1):
            policy = self.get_policy(name)
            if policy is not None:
                lines.append(f"{i}. {policy.summary()}")

        return "\n".join(lines)

    def to_standard_result(self) -> EstimationResult:
        """Convert to standard EstimationResult for compatibility."""
        # Extract first policy's EIF components if available
        first_policy_meta = self.policies[0].metadata
        eif_components = first_policy_meta.get("eif_components", None)

        return EstimationResult(
            estimator_type=self.estimator_type,
            v_hat=self.get_estimates(),
            se=self.get_standard_errors(),
            n=self.n_samples,
            eif_components=eif_components,
            metadata=self.global_metadata,
        )


def create_multi_policy_result(
    estimates: List[UncertaintyAwareEstimate],
    policy_names: Optional[List[str]] = None,
    n_samples: int = 0,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    estimator_type: str = "UncertaintyAwareDRCPO",
) -> MultiPolicyUncertaintyResult:
    """Create multi-policy result from individual estimates.

    Args:
        estimates: Uncertainty-aware estimates for each policy
        policy_names: Names for each policy
        n_samples: Total number of samples
        metadata_list: Per-policy metadata
        estimator_type: Type of estimator used

    Returns:
        MultiPolicyUncertaintyResult with all information
    """
    if not estimates:
        raise ValueError("At least one estimate required")

    # Generate default names if not provided
    if policy_names is None:
        policy_names = [f"Policy_{i}" for i in range(len(estimates))]

    if len(policy_names) != len(estimates):
        raise ValueError("Number of names must match number of estimates")

    # Create metadata list if not provided
    if metadata_list is None:
        metadata_list = [{}] * len(estimates)

    # Create policy results
    policies = [
        PolicyResult(name=name, estimate=est, metadata=meta)
        for name, est, meta in zip(policy_names, estimates, metadata_list)
    ]

    return MultiPolicyUncertaintyResult(
        policies=policies,
        n_samples=n_samples,
        estimator_type=estimator_type,
    )
