"""Data schemas for uncertainty-aware evaluation.

All schemas in this module treat uncertainty as a first-class citizen.
Every score has an associated variance (which can be 0 for deterministic judges).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class JudgeScore:
    """Score from a judge with uncertainty quantification.

    Attributes:
        mean: Expected score value (0-1 range)
        variance: Uncertainty in the score (0 for deterministic judges)
    """

    mean: float
    variance: float

    def __post_init__(self) -> None:
        """Validate score values."""
        if not 0 <= self.mean <= 1:
            raise ValueError(f"Score mean {self.mean} must be in [0, 1]")
        if self.variance < 0:
            raise ValueError(f"Variance {self.variance} cannot be negative")
        if self.variance > 0.25:  # Max variance for [0,1] bounded variable
            raise ValueError(f"Variance {self.variance} too large for [0,1] score")


@dataclass
class CalibratedReward:
    """Calibrated reward with variance scaling.

    Attributes:
        value: Calibrated reward value (0-1 range)
        variance: Calibrated variance (scaled by gamma)
        gamma: Variance scale factor applied
        raw_score: Original judge score before calibration
        raw_variance: Original variance before calibration
    """

    value: float
    variance: float
    gamma: float
    raw_score: float
    raw_variance: float

    @property
    def standard_deviation(self) -> float:
        """Compute standard deviation from variance."""
        return float(np.sqrt(self.variance))


@dataclass
class VarianceDecomposition:
    """Decomposition of total variance into components.

    Attributes:
        total: Total variance
        eif: Variance from efficient influence function
        judge: Variance from judge uncertainty
        eif_pct: Percentage contribution from EIF
        judge_pct: Percentage contribution from judge
    """

    total: float
    eif: float
    judge: float
    eif_pct: float
    judge_pct: float

    def __post_init__(self) -> None:
        """Validate percentages sum to 100."""
        if abs(self.eif_pct + self.judge_pct - 100.0) > 0.1:
            raise ValueError(
                f"Percentages must sum to 100, got {self.eif_pct + self.judge_pct}"
            )


@dataclass
class UncertaintyAwareEstimate:
    """Final estimate with uncertainty quantification.

    Attributes:
        value: Point estimate
        se: Standard error including judge uncertainty
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        variance_decomposition: Breakdown of variance sources
        effective_sample_size: ESS after any weight adjustments
        shrinkage_applied: Whether variance shrinkage was used
        shrinkage_lambda: Shrinkage parameter if applied
    """

    value: float
    se: float
    ci_lower: float
    ci_upper: float
    variance_decomposition: VarianceDecomposition
    effective_sample_size: float
    shrinkage_applied: bool = False
    shrinkage_lambda: Optional[float] = None

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval."""
        return self.ci_upper - self.ci_lower

    def summary(self) -> str:
        """Human-readable summary of the estimate."""
        shrinkage_info = (
            f" (λ={self.shrinkage_lambda:.3f})" if self.shrinkage_applied else ""
        )
        return (
            f"Estimate: {self.value:.4f} ± {self.se:.4f}\n"
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"ESS: {self.effective_sample_size:.1f}{shrinkage_info}\n"
            f"Variance: {self.variance_decomposition.eif_pct:.1f}% EIF, "
            f"{self.variance_decomposition.judge_pct:.1f}% judge"
        )


@dataclass
class UncertaintyDiagnostics:
    """Detailed diagnostics for uncertainty contributions.

    Attributes:
        per_sample_contributions: Variance contribution from each sample
        high_variance_samples: Indices of samples with high variance contribution
        concentration_ratio: Fraction of variance from top 10% of samples
        gamma_calibration: Variance scale factor from calibration
        warnings: List of diagnostic warnings
    """

    per_sample_contributions: np.ndarray
    high_variance_samples: List[int]
    concentration_ratio: float
    gamma_calibration: float
    warnings: List[str]

    def plot_contributions(self) -> None:
        """Plot variance contributions (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            sorted_contributions = np.sort(self.per_sample_contributions)[::-1]
            plt.plot(sorted_contributions)
            plt.xlabel("Sample rank")
            plt.ylabel("Variance contribution")
            plt.title("Per-sample variance contributions")
            plt.yscale("log")
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
