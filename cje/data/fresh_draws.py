"""Data models for fresh draws used in DR estimation."""

from typing import Dict, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field, field_validator


class FreshDrawSample(BaseModel):
    """A single fresh draw sample for DR estimation.

    Represents a fresh response sampled from a target policy,
    evaluated by the judge.
    """

    prompt_id: str = Field(..., description="ID to align with logged data")
    target_policy: str = Field(..., description="Policy that generated this response")
    judge_score: float = Field(..., ge=0, le=1, description="Judge evaluation score")
    response: Optional[str] = Field(None, description="Generated response (optional)")
    draw_idx: int = Field(
        ..., ge=0, description="Draw index for this prompt (0, 1, 2...)"
    )
    fold_id: Optional[int] = Field(
        None, description="CV fold assignment (should match logged data)"
    )

    @field_validator("judge_score")
    def validate_judge_score(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"Judge score must be in [0, 1], got {v}")
        return v


class FreshDrawDataset(BaseModel):
    """Collection of fresh draws for a target policy.

    Contains pre-generated fresh samples from a target policy,
    evaluated by a judge, for use in DR estimation.
    """

    target_policy: str = Field(..., description="Target policy name")
    draws_per_prompt: int = Field(..., ge=1, description="Number of draws per prompt")
    samples: List[FreshDrawSample] = Field(..., min_length=1)

    @field_validator("samples")
    def validate_samples(
        cls, v: List[FreshDrawSample], info: Any
    ) -> List[FreshDrawSample]:
        """Ensure samples are consistent."""
        if "target_policy" in info.data:
            policy = info.data["target_policy"]
            for sample in v:
                if sample.target_policy != policy:
                    raise ValueError(
                        f"Sample has policy '{sample.target_policy}' "
                        f"but dataset is for '{policy}'"
                    )
        return v

    def get_prompt_ids(self) -> List[str]:
        """Get unique prompt IDs in dataset."""
        return sorted(set(s.prompt_id for s in self.samples))

    def get_scores_for_prompt_id(self, prompt_id: str) -> np.ndarray:
        """Get judge scores for a specific prompt.

        Args:
            prompt_id: The prompt ID to get scores for

        Returns:
            Array of judge scores for this prompt, sorted by draw_idx
        """
        # Sort by draw_idx for reproducibility
        matching_samples = sorted(
            [s for s in self.samples if s.prompt_id == prompt_id],
            key=lambda s: s.draw_idx,
        )

        if not matching_samples:
            raise ValueError(f"No samples found for prompt_id '{prompt_id}'")

        if len(matching_samples) != self.draws_per_prompt:
            raise ValueError(
                f"Expected {self.draws_per_prompt} draws for prompt '{prompt_id}', "
                f"found {len(matching_samples)}"
            )

        return np.array([s.judge_score for s in matching_samples])

    def get_samples_for_prompt_id(self, prompt_id: str) -> List[FreshDrawSample]:
        """Get all samples for a specific prompt.

        Args:
            prompt_id: The prompt ID to get samples for

        Returns:
            List of samples for this prompt
        """
        samples = [s for s in self.samples if s.prompt_id == prompt_id]

        if not samples:
            raise ValueError(f"No samples found for prompt_id '{prompt_id}'")

        # Sort by draw_idx to ensure consistent ordering
        return sorted(samples, key=lambda s: s.draw_idx)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert dataset to arrays for efficient computation.

        Returns:
            Dict with 'prompt_ids' and 'judge_scores' arrays
        """
        # Sort samples by (prompt_id, draw_idx) for consistent ordering
        sorted_samples = sorted(self.samples, key=lambda s: (s.prompt_id, s.draw_idx))

        prompt_ids = []
        judge_scores = []

        for sample in sorted_samples:
            prompt_ids.append(sample.prompt_id)
            judge_scores.append(sample.judge_score)

        return {
            "prompt_ids": np.array(prompt_ids),
            "judge_scores": np.array(judge_scores),
        }

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics for the dataset."""
        scores = np.array([s.judge_score for s in self.samples])
        unique_prompts = self.get_prompt_ids()

        return {
            "target_policy": self.target_policy,
            "n_samples": len(self.samples),
            "n_prompts": len(unique_prompts),
            "draws_per_prompt": self.draws_per_prompt,
            "judge_score_mean": float(scores.mean()),
            "judge_score_std": float(scores.std()),
            "judge_score_min": float(scores.min()),
            "judge_score_max": float(scores.max()),
        }
