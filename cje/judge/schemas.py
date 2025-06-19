"""Pydantic schemas for structured judge outputs."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import json


class JudgeScore(BaseModel):
    """Basic structured score from a judge."""

    score: float = Field(
        ge=0,
        le=10,
        description="Numeric score from 0-10 evaluating the response quality",
    )

    @field_validator("score")
    def normalize_score(cls, v: float) -> float:
        """Ensure score is within 0-1 range for backwards compatibility."""
        # If score is already 0-1, keep it
        if 0 <= v <= 1:
            return v
        # If score is 0-10, normalize to 0-1
        elif 0 <= v <= 10:
            return v / 10
        else:
            raise ValueError(f"Score {v} is out of valid range")


class JudgeEvaluationMetadata(BaseModel):
    """OpenAI-compatible metadata object with explicit schema."""

    # Define common metadata fields explicitly instead of using Dict[str, Any]
    # This ensures OpenAI compatibility by avoiding additionalProperties issues

    template_used: Optional[str] = Field(
        default=None, description="Judge template used"
    )
    provider: Optional[str] = Field(default=None, description="API provider used")
    model_name: Optional[str] = Field(default=None, description="Model name used")
    response_time_ms: Optional[float] = Field(
        default=None, description="Response time in milliseconds"
    )
    tokens_used: Optional[int] = Field(default=None, description="Total tokens used")

    class Config:
        # Explicitly forbid additional properties for OpenAI compatibility
        extra = "forbid"


class JudgeEvaluation(BaseModel):
    """Comprehensive structured evaluation from a judge."""

    score: float = Field(
        ge=0,
        le=10,
        description="Numeric score from 0-10 evaluating the response quality",
    )

    reasoning: str = Field(
        min_length=10,
        description="Detailed explanation for why this score was assigned",
    )

    confidence: float = Field(
        ge=0,
        le=1,
        default=0.8,
        description="Judge's confidence level in this evaluation (0-1)",
    )

    key_strengths: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="List of up to 5 specific strengths identified in the response",
    )

    key_weaknesses: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="List of up to 5 weaknesses or areas for improvement",
    )

    improvement_suggestions: Optional[str] = Field(
        default=None, description="Specific suggestions for how to improve the response"
    )

    metadata: JudgeEvaluationMetadata = Field(
        default_factory=JudgeEvaluationMetadata,
        description="Additional metadata about the evaluation",
    )

    @field_validator("score")
    def normalize_score(cls, v: float) -> float:
        """Ensure score is within 0-1 range for backwards compatibility."""
        # If score is already 0-1, keep it
        if 0 <= v <= 1:
            return v
        # If score is 0-10, normalize to 0-1
        elif 0 <= v <= 10:
            return v / 10
        else:
            raise ValueError(f"Score {v} is out of valid range")

    @field_validator("reasoning")
    def check_reasoning_quality(cls, v: str) -> str:
        """Ensure reasoning is substantive."""
        if len(v.split()) < 10:
            raise ValueError("Reasoning must be at least 10 words")
        return v

    class Config:
        # Explicitly forbid additional properties for OpenAI compatibility
        extra = "forbid"


class DetailedJudgeEvaluation(BaseModel):
    """Most detailed evaluation schema for advanced use cases."""

    overall_score: float = Field(ge=0, le=10, description="Overall score from 0-10")

    accuracy_score: float = Field(
        ge=0, le=10, description="Score for factual accuracy and correctness"
    )

    completeness_score: float = Field(
        ge=0, le=10, description="Score for completeness and coverage of the topic"
    )

    clarity_score: float = Field(
        ge=0, le=10, description="Score for clarity and coherence of expression"
    )

    relevance_score: float = Field(
        ge=0, le=10, description="Score for relevance to the original question/context"
    )

    reasoning: str = Field(
        min_length=50, description="Comprehensive explanation of the evaluation"
    )

    strengths: List[str] = Field(
        min_length=1, max_length=5, description="Key strengths of the response"
    )

    weaknesses: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Key weaknesses or areas for improvement",
    )

    suggestions: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Specific suggestions for improvement",
    )

    confidence: float = Field(
        ge=0, le=1, default=0.8, description="Confidence in this evaluation"
    )

    @field_validator(
        "overall_score",
        "accuracy_score",
        "completeness_score",
        "clarity_score",
        "relevance_score",
    )
    def normalize_scores(cls, v: float) -> float:
        """Normalize all scores to 0-1 range."""
        if 0 <= v <= 1:
            return v
        elif 0 <= v <= 10:
            return v / 10
        else:
            raise ValueError(f"Score {v} is out of valid range")


# Type alias for any judge evaluation
JudgeResult = Union[
    JudgeScore,
    JudgeEvaluation,
    DetailedJudgeEvaluation,
]
