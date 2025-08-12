"""Data models for CJE using Pydantic."""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import numpy as np


class LogProbStatus(Enum):
    """Status of log probability computation."""

    SUCCESS = "success"
    API_ERROR = "api_error"
    TOKEN_BOUNDARY_ERROR = "token_boundary_error"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    EMPTY_RESPONSE = "empty_response"


class LogProbResult(BaseModel):
    """Result of log probability computation with explicit error handling."""

    value: Optional[float] = Field(
        None, description="Log probability value if successful"
    )
    status: LogProbStatus = Field(
        LogProbStatus.API_ERROR, description="Computation status"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @property
    def is_valid(self) -> bool:
        """Check if computation succeeded."""
        return self.status == LogProbStatus.SUCCESS and self.value is not None


class Sample(BaseModel):
    """A single sample for CJE analysis."""

    prompt: str = Field(..., description="Input prompt/context")
    response: str = Field(..., description="Generated response")
    reward: Optional[float] = Field(
        None, ge=0, le=1, description="Calibrated reward [0,1]"
    )
    base_policy_logprob: Optional[float] = Field(
        None, description="Log prob under base policy"
    )
    target_policy_logprobs: Dict[str, Optional[float]] = Field(
        ..., description="Log probs under target policies (None for failures)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata"
    )

    @field_validator("base_policy_logprob")
    def validate_base_policy_logprob(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v > 0:
            raise ValueError(f"Log probability must be <= 0, got {v}")
        return v

    @field_validator("target_policy_logprobs")
    def validate_target_policy_logprobs(
        cls, v: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        for policy, logprob in v.items():
            if logprob is not None and logprob > 0:
                raise ValueError(
                    f"Log probability for {policy} must be <= 0, got {logprob}"
                )
        return v

    def get_importance_weight(self, target_policy: str) -> Optional[float]:
        """Compute importance weight for a target policy."""
        if self.base_policy_logprob is None:
            return None
        target_lp = self.target_policy_logprobs.get(target_policy)
        if target_lp is None:
            return None
        return float(np.exp(target_lp - self.base_policy_logprob))


class Dataset(BaseModel):
    """A dataset for CJE analysis.

    This is a pure data container following the Single Responsibility Principle.
    For loading data, use DatasetFactory or DatasetLoader.
    """

    samples: List[Sample] = Field(..., min_length=1)
    target_policies: List[str] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_policies")
    def validate_policies_exist(cls, v: List[str], info: Any) -> List[str]:
        """Ensure target policies exist in samples."""
        if "samples" in info.data:
            all_policies = set()
            for sample in info.data["samples"]:
                all_policies.update(sample.target_policy_logprobs.keys())

            missing = set(v) - all_policies
            if missing:
                raise ValueError(f"Target policies not found in data: {missing}")
        return v

    def filter_valid_samples(self, target_policy: str) -> List[Sample]:
        """Get samples with valid data for a specific target policy."""
        valid_samples = []
        for sample in self.samples:
            if (
                sample.base_policy_logprob is not None
                and sample.target_policy_logprobs.get(target_policy) is not None
            ):
                valid_samples.append(sample)
        return valid_samples

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        rewards = [s.reward for s in self.samples]
        valid_counts = {policy: 0 for policy in self.target_policies}

        for sample in self.samples:
            for policy in self.target_policies:
                if sample.get_importance_weight(policy) is not None:
                    valid_counts[policy] += 1

        return {
            "n_samples": self.n_samples,
            "target_policies": self.target_policies,
            "reward_mean": np.mean(rewards),
            "reward_std": np.std(rewards),
            "valid_samples_per_policy": valid_counts,
        }


class EstimationResult(BaseModel):
    """Result from a CJE estimator."""

    estimates: np.ndarray = Field(..., description="Point estimates for each policy")
    standard_errors: np.ndarray = Field(..., description="Standard errors")
    n_samples_used: Dict[str, int] = Field(..., description="Valid samples per policy")
    method: str = Field(..., description="Estimation method used")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Compute confidence intervals."""
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        lower = self.estimates - z * self.standard_errors
        upper = self.estimates + z * self.standard_errors
        return lower, upper

    def best_policy(self) -> int:
        """Get index of best policy by point estimate."""
        return int(np.argmax(self.estimates))

    def compare_policies(
        self, idx1: int, idx2: int, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Compare two policies."""
        diff = self.estimates[idx1] - self.estimates[idx2]
        se_diff = np.sqrt(
            self.standard_errors[idx1] ** 2 + self.standard_errors[idx2] ** 2
        )
        z_score = diff / se_diff if se_diff > 0 else 0

        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            "difference": diff,
            "se_difference": se_diff,
            "z_score": z_score,
            "p_value": p_value,
            "significant": p_value < alpha,
        }


class WeightCalibrationConfig(BaseModel):
    """Configuration for weight calibration."""

    k_folds: int = Field(5, ge=2, description="Number of cross-fitting folds")
    clip_weight: Optional[float] = Field(
        None, description="Maximum weight value (None = no clipping)"
    )
    target_mean: float = Field(1.0, gt=0, description="Target mean for calibration")
    random_seed: int = Field(42, description="Random seed for reproducibility")
