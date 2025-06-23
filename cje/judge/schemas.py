"""Unified judge score schemas with built-in uncertainty support.

This module provides the single source of truth for judge scores,
replacing the dual system of legacy float scores and uncertainty scores.
"""

from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field, field_validator, model_validator


class JudgeScore(BaseModel):
    """Unified judge score with mandatory uncertainty quantification.

    All judges now return structured scores with mean and variance.
    For deterministic judges, variance is simply 0.

    Attributes:
        mean: The point estimate of the score (0-1 range)
        variance: The uncertainty/variance of the score (0-0.25 range)
    """

    mean: float = Field(
        ..., ge=0, le=1, description="The mean/expected score value in [0, 1] range"
    )
    variance: float = Field(
        default=0.0,
        ge=0,
        le=0.25,  # Max variance for [0,1] bounded variable
        description="The variance/uncertainty of the score (0 for deterministic judges)",
    )

    @field_validator("mean", mode="before")
    @classmethod
    def normalize_mean(cls, v: Union[float, int]) -> float:
        """Normalize scores from 0-10 range to 0-1 range if needed.

        Maintains backward compatibility with judges that return 0-10 scores.
        """
        v = float(v)
        if 0 <= v <= 1:
            return v
        elif 0 <= v <= 10:
            return v / 10
        else:
            raise ValueError(f"Score mean {v} is out of valid range [0, 10]")

    @model_validator(mode="after")
    def validate_variance_bounds(self) -> "JudgeScore":
        """Ensure variance is reasonable for the mean value."""
        # For scores near boundaries, variance should be limited
        max_var = self.mean * (1 - self.mean)  # Variance of Bernoulli
        if self.variance > max_var:
            # Clip to maximum possible variance
            self.variance = max_var
        return self

    @property
    def se(self) -> float:
        """Standard error (square root of variance)."""
        return float(self.variance**0.5)

    @property
    def score(self) -> float:
        """Get score in 0-10 range for backward compatibility and convenience."""
        return self.mean * 10

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval assuming normal approximation."""
        import scipy.stats

        z = scipy.stats.norm.ppf(1 - alpha / 2)
        margin = z * self.se
        return (max(0, self.mean - margin), min(1, self.mean + margin))


class JudgeScoreWithCI(JudgeScore):
    """Judge score with explicit confidence interval fields.

    This schema is used when we want the judge to provide
    confidence intervals directly rather than just variance.
    """

    ci_lower: float = Field(
        ...,
        ge=0,
        le=10,
        description="Lower bound of 95% confidence interval (0-10 scale)",
    )
    ci_upper: float = Field(
        ...,
        ge=0,
        le=10,
        description="Upper bound of 95% confidence interval (0-10 scale)",
    )

    @field_validator("ci_lower", "ci_upper", mode="before")
    @classmethod
    def validate_ci_bounds(cls, v: Union[float, int]) -> float:
        """Ensure CI bounds are valid."""
        return float(v)

    @model_validator(mode="after")
    def calculate_variance_from_ci(self) -> "JudgeScoreWithCI":
        """Calculate variance from the confidence interval.

        Assumes 95% CI = mean ± 1.96*σ
        Therefore: σ = (ci_upper - ci_lower) / (2 * 1.96)
        Variance = σ²
        """
        # Ensure CI contains the mean score
        score_10_scale = self.mean * 10
        if not (self.ci_lower <= score_10_scale <= self.ci_upper):
            # Adjust CI to contain the score
            self.ci_lower = min(self.ci_lower, score_10_scale)
            self.ci_upper = max(self.ci_upper, score_10_scale)

        # Calculate variance from CI width
        ci_range = self.ci_upper - self.ci_lower
        sd_10_scale = ci_range / 3.92  # 2 * 1.96 = 3.92
        sd_01_scale = sd_10_scale / 10  # Convert to 0-1 scale
        self.variance = sd_01_scale**2

        return self


class JudgeEvaluation(JudgeScore):
    """Extended evaluation with reasoning and metadata.

    Backward compatible with legacy JudgeEvaluation but now
    includes uncertainty quantification.
    """

    reasoning: Optional[str] = Field(None, description="Explanation of the score")
    confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Judge's self-reported confidence (deprecated - use variance instead)",
    )
    strengths: List[str] = Field(
        default_factory=list, description="Identified strengths in the response"
    )
    weaknesses: List[str] = Field(
        default_factory=list, description="Identified weaknesses in the response"
    )

    @model_validator(mode="after")
    def convert_confidence_to_variance(self) -> "JudgeEvaluation":
        """Convert legacy confidence to variance if variance not set."""
        if self.confidence is not None and self.variance == 0.0:
            # High confidence = low variance
            # Map confidence [0,1] to variance [0.05, 0.0]
            self.variance = 0.05 * (1 - self.confidence)
        return self


class DetailedJudgeEvaluation(JudgeEvaluation):
    """Comprehensive evaluation with sub-dimensions.

    Now includes uncertainty for each sub-score.
    """

    relevance_score: Optional[JudgeScore] = Field(
        None, description="How relevant the response is to the question"
    )
    accuracy_score: Optional[JudgeScore] = Field(
        None, description="Factual accuracy of the response"
    )
    clarity_score: Optional[JudgeScore] = Field(
        None, description="Clarity and coherence of the response"
    )
    completeness_score: Optional[JudgeScore] = Field(
        None, description="How complete/comprehensive the response is"
    )

    @model_validator(mode="after")
    def compute_overall_from_subscores(self) -> "DetailedJudgeEvaluation":
        """Compute overall score from sub-scores if not provided."""
        if self.mean == 0.0 and self.variance == 0.0:
            # Compute from available sub-scores
            sub_scores = [
                s
                for s in [
                    self.relevance_score,
                    self.accuracy_score,
                    self.clarity_score,
                    self.completeness_score,
                ]
                if s is not None
            ]

            if sub_scores:
                # Average of means
                self.mean = sum(s.mean for s in sub_scores) / len(sub_scores)
                # Average of variances (assuming independence)
                self.variance = sum(s.variance for s in sub_scores) / (
                    len(sub_scores) ** 2
                )

        return self


# Type alias for any judge result
JudgeResult = Union[JudgeScore, JudgeEvaluation, DetailedJudgeEvaluation]
