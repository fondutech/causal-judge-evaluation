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

    def __float__(self) -> float:
        """Convert to float for backward compatibility.

        Returns the mean value when cast to float.
        """
        return self.mean

    @property
    def value(self) -> float:
        """Alias for mean, matching uncertainty module interface."""
        return self.mean

    @property
    def se(self) -> float:
        """Standard error (square root of variance)."""
        return float(self.variance**0.5)

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval assuming normal approximation."""
        import scipy.stats

        z = scipy.stats.norm.ppf(1 - alpha / 2)
        margin = z * self.se
        return (max(0, self.mean - margin), min(1, self.mean + margin))


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


# Backward compatibility helpers
def score_to_float(score: Union[float, JudgeScore]) -> float:
    """Convert any score type to float for legacy code."""
    if isinstance(score, (int, float)):
        return float(score)
    return float(score.mean)


def scores_to_floats(scores: List[Union[float, JudgeScore]]) -> List[float]:
    """Convert list of scores to floats for legacy code."""
    return [score_to_float(s) for s in scores]


def float_to_score(value: float, variance: float = 0.0) -> JudgeScore:
    """Convert float to JudgeScore for unified interface."""
    return JudgeScore(mean=value, variance=variance)


def floats_to_scores(
    values: List[float], variances: Optional[List[float]] = None
) -> List[JudgeScore]:
    """Convert list of floats to JudgeScores."""
    if variances is None:
        variances = [0.0] * len(values)
    return [JudgeScore(mean=v, variance=var) for v, var in zip(values, variances)]
