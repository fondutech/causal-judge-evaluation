"""Result objects for CJE estimators.

These provide a clean, intuitive API for accessing estimation results
without needing to remember dictionary keys.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from scipy import stats
from dataclasses import dataclass, field


@dataclass
class EstimationResult:
    """Base class for estimation results."""

    # Core results
    v_hat: np.ndarray
    se: np.ndarray
    n: int

    # Optional detailed results
    eif_components: Optional[np.ndarray] = None
    pairwise_p_values: Optional[np.ndarray] = None
    pairwise_z_scores: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None

    # Metadata
    estimator_type: str = "unknown"
    n_policies: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and setup derived attributes."""
        self.v_hat = np.asarray(self.v_hat)
        self.se = np.asarray(self.se)
        self.n_policies = len(self.v_hat)

    @property
    def estimates(self) -> np.ndarray:
        """Alias for v_hat for more intuitive access."""
        return self.v_hat

    @property
    def standard_errors(self) -> np.ndarray:
        """Alias for se for more intuitive access."""
        return self.se

    def confidence_interval(self, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals at specified level.

        Args:
            level: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        z = stats.norm.ppf((1 + level) / 2)
        ci_lower = self.v_hat - z * self.se
        ci_upper = self.v_hat + z * self.se
        return ci_lower, ci_upper

    def confidence_intervals(
        self, level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for confidence_interval() for backward compatibility.

        Args:
            level: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        return self.confidence_interval(level)

    def best_policy(self) -> int:
        """Get index of best performing policy."""
        return int(np.argmax(self.v_hat))

    def rank_policies(self) -> List[int]:
        """Get policies ranked from best to worst."""
        return list(np.argsort(self.v_hat)[::-1])

    def compare_policies(
        self, i: int, j: int, correction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare two policies.

        Args:
            i: Index of first policy
            j: Index of second policy
            correction: Multiple testing correction ('bonferroni', 'holm', None)

        Returns:
            Dict with difference, SE, p-value, and significance
        """
        diff = self.v_hat[i] - self.v_hat[j]

        if self.covariance_matrix is not None:
            se_diff = np.sqrt(
                self.covariance_matrix[i, i]
                + self.covariance_matrix[j, j]
                - 2 * self.covariance_matrix[i, j]
            )
        else:
            # Conservative estimate if no covariance
            se_diff = np.sqrt(self.se[i] ** 2 + self.se[j] ** 2)

        z_score = diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Apply correction if requested
        if correction and self.n_policies > 2:
            n_comparisons = self.n_policies * (self.n_policies - 1) / 2
            if correction == "bonferroni":
                p_value *= n_comparisons
            # Add more corrections as needed

        return {
            "difference": float(diff),
            "se_difference": float(se_diff),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "favors": "policy_i" if diff > 0 else "policy_j",
        }

    def bootstrap_confidence_intervals(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Bayesian bootstrap confidence intervals using influence functions.

        This method provides non-parametric confidence intervals that remain valid
        when sample sizes are small or when the asymptotic normality assumption
        of analytical SEs may be violated.

        Args:
            confidence_level: Confidence level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap replicates (default 1000)
            seed: Random seed for reproducibility

        Returns:
            Dict containing:
                - 'ci_lower': Lower confidence bounds (K,)
                - 'ci_upper': Upper confidence bounds (K,)
                - 'bootstrap_samples': All bootstrap replicates (n_bootstrap, K)
                - 'bootstrap_se': Bootstrap standard errors (K,)

        Raises:
            ValueError: If influence functions are not available

        Note:
            This uses the Bayesian bootstrap (Dirichlet weights) which is
            equivalent to the non-parametric bootstrap but computationally
            more efficient since it only requires reweighting the existing
            influence function values.
        """
        if self.eif_components is None:
            raise ValueError(
                "Influence functions not available for bootstrap. "
                "This estimator may not support bootstrap inference."
            )

        np.random.seed(seed)
        n, K = self.eif_components.shape
        bootstrap_estimates: List[np.ndarray] = []

        for _ in range(n_bootstrap):
            # Draw Dirichlet weights (equivalent to exponential(1) then normalize)
            # This is the Bayesian bootstrap: each observation gets a random weight
            weights = np.random.exponential(1.0, size=n)
            weights = weights / weights.mean()  # Normalize to mean 1

            # Compute bootstrap replicate: Σ dᵢ φᵢ
            # This reweights the influence function to get a bootstrap estimate
            v_boot = np.average(self.eif_components, weights=weights, axis=0)
            bootstrap_estimates.append(v_boot)

        bootstrap_estimates_array = np.array(
            bootstrap_estimates
        )  # Shape: (n_bootstrap, K)

        # Compute percentile confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_estimates_array, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(
            bootstrap_estimates_array, 100 * (1 - alpha / 2), axis=0
        )

        return {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bootstrap_samples": bootstrap_estimates_array,
            "bootstrap_se": np.std(bootstrap_estimates_array, axis=0, ddof=1),
        }

    def summary(self, include_bootstrap: bool = False) -> str:
        """Get a human-readable summary of results.

        Args:
            include_bootstrap: Whether to include bootstrap CIs alongside analytical CIs
        """
        lines = [
            f"Estimation Results ({self.estimator_type})",
            f"Number of policies: {self.n_policies}",
            f"Sample size: {self.n}",
            "",
        ]

        # Add estimates with CIs
        ci_lower, ci_upper = self.confidence_interval()

        # Try to get bootstrap CIs if requested and available
        bootstrap_available = include_bootstrap and self.eif_components is not None
        if bootstrap_available:
            try:
                bootstrap_ci = self.bootstrap_confidence_intervals(seed=42)
                boot_lower = bootstrap_ci["ci_lower"]
                boot_upper = bootstrap_ci["ci_upper"]
                lines.append("Analytical vs Bootstrap Confidence Intervals:")
                lines.append("")
            except Exception:
                bootstrap_available = False

        for i in range(self.n_policies):
            if bootstrap_available:
                lines.append(f"Policy {i}: {self.v_hat[i]:.3f} ± {self.se[i]:.3f}")
                lines.append(f"  Analytical CI: [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]")
                lines.append(
                    f"  Bootstrap CI:  [{boot_lower[i]:.3f}, {boot_upper[i]:.3f}]"
                )
            else:
                lines.append(
                    f"Policy {i}: {self.v_hat[i]:.3f} ± {self.se[i]:.3f} "
                    f"[{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]"
                )

        # Add best policy
        lines.append("")
        lines.append(f"Best policy: Policy {self.best_policy()}")

        # Add pairwise comparisons if available
        if self.n_policies > 1:
            lines.append("")
            lines.append("Pairwise comparisons (p-values):")
            for i in range(self.n_policies):
                for j in range(i + 1, self.n_policies):
                    comp = self.compare_policies(i, j)
                    sig = "*" if comp["significant"] else ""
                    lines.append(f"  Policy {i} vs {j}: p={comp['p_value']:.3f}{sig}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"EstimationResult(n_policies={self.n_policies}, "
            f"best={self.best_policy()}, "
            f"v_hat={self.v_hat[self.best_policy()]:.3f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility/serialization."""
        ci_lower, ci_upper = self.confidence_interval()
        result = {
            "v_hat": self.v_hat.tolist(),
            "se": self.se.tolist(),
            "n": self.n,
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
            "estimator_type": self.estimator_type,
            "n_policies": self.n_policies,
        }

        # Add optional fields if present
        if self.eif_components is not None:
            result["eif_components"] = self.eif_components.tolist()
        if self.pairwise_p_values is not None:
            result["pairwise_p_values"] = self.pairwise_p_values.tolist()
        if self.pairwise_z_scores is not None:
            result["pairwise_z_scores"] = self.pairwise_z_scores.tolist()
        if self.covariance_matrix is not None:
            result["covariance_matrix"] = self.covariance_matrix.tolist()

        result.update(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EstimationResult":
        """Create from dictionary for compatibility."""
        # Extract core fields
        v_hat = np.array(data["v_hat"])
        se = np.array(data["se"])
        n = data["n"]

        # Extract optional fields
        kwargs = {
            "estimator_type": data.get("estimator_type", "unknown"),
            "n_policies": data.get("n_policies", len(v_hat)),
        }

        if "eif_components" in data:
            kwargs["eif_components"] = np.array(data["eif_components"])
        if "pairwise_p_values" in data:
            kwargs["pairwise_p_values"] = np.array(data["pairwise_p_values"])
        if "pairwise_z_scores" in data:
            kwargs["pairwise_z_scores"] = np.array(data["pairwise_z_scores"])
        if "covariance_matrix" in data:
            kwargs["covariance_matrix"] = np.array(data["covariance_matrix"])

        # Store any extra fields as metadata
        known_fields = {
            "v_hat",
            "se",
            "n",
            "ci_lower",
            "ci_upper",
            "estimator_type",
            "n_policies",
            "eif_components",
            "pairwise_p_values",
            "pairwise_z_scores",
            "covariance_matrix",
        }
        metadata = {k: v for k, v in data.items() if k not in known_fields}
        if metadata:
            kwargs["metadata"] = metadata

        return cls(v_hat=v_hat, se=se, n=n, **kwargs)
