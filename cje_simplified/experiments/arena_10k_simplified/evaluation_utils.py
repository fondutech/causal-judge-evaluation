"""
Shared evaluation utilities for judges and oracles.

This module contains reusable implementations for scoring responses
using LLM-based judges and oracles with structured outputs.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm import tqdm

# For structured outputs with LangChain
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model


# Data models
@dataclass
class JudgeScore:
    """Result from scoring a single response."""

    score: float  # Score in [0, 1] (normalized from 0-100)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchJudgeResult:
    """Result from batch scoring."""

    scores: List[float]
    metadata: Optional[List[Dict[str, Any]]] = None
    mean_score: float = 0.0
    std_score: float = 0.0

    def __post_init__(self) -> None:
        """Compute statistics."""
        if self.scores:
            self.mean_score = np.mean(self.scores)
            self.std_score = np.std(self.scores)


# Pydantic model for structured outputs
class EvaluationResponse(BaseModel):
    """Structured response from the evaluator. Reason first, then score."""

    reasoning: str = Field(description="Explanation of the evaluation")
    score: float = Field(
        description="Score from 0 to 100 based on response quality",
        ge=0.0,
        le=100.0,
    )


# Evaluator class
class FireworksEvaluator:
    """Fireworks-based evaluator for scoring AI responses."""

    def __init__(
        self,
        model: str,
        system_prompt: str = "You are an expert evaluator that assesses AI response quality.",
        user_prompt_template: str = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = (
            user_prompt_template or self._default_prompt_template()
        )
        self.temperature = temperature
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY required")

        # Initialize LangChain model with Fireworks
        os.environ["FIREWORKS_API_KEY"] = self.api_key
        self.llm = init_chat_model(
            model,
            model_provider="fireworks",
            temperature=temperature,
        )

        # Create structured LLM
        self.structured_llm = self.llm.with_structured_output(EvaluationResponse)

    def _default_prompt_template(self) -> str:
        """Default prompt template."""
        return """Evaluate the AI response below.

<user_prompt>
{prompt}
</user_prompt>

<ai_response>
{response}
</ai_response>

Rate the response quality from 0 to 100, considering relevance, helpfulness, clarity, and accuracy."""

    def score(self, prompt: str, response: str) -> JudgeScore:
        """Score a single response."""
        # Format user message from template
        user_message = self.user_prompt_template.format(
            prompt=prompt, response=response
        )

        try:
            # Get structured response
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ]

            result: EvaluationResponse = self.structured_llm.invoke(messages)

            # Normalize score from 0-100 to 0-1
            normalized_score = result.score / 100.0

            return JudgeScore(
                score=normalized_score,
                metadata={
                    "reasoning": result.reasoning,
                    "model": self.model,
                    "raw_score": result.score,  # Keep raw 0-100 score
                },
            )

        except Exception as e:
            # Return neutral score on error
            return JudgeScore(
                score=0.5, metadata={"error": str(e), "model": self.model}
            )

    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
        show_progress: bool = True,
        desc: str = "Scoring",
    ) -> BatchJudgeResult:
        """Score a batch of responses."""
        if len(prompts) != len(responses):
            raise ValueError(f"Length mismatch: {len(prompts)} != {len(responses)}")

        scores = []
        metadata = []

        iterator = zip(prompts, responses)
        if show_progress:
            iterator = tqdm(iterator, total=len(prompts), desc=desc)

        for prompt, response in iterator:
            result = self.score(prompt, response)
            scores.append(result.score)
            if result.metadata:
                metadata.append(result.metadata)

        return BatchJudgeResult(scores=scores, metadata=metadata if metadata else None)


# Default models
DEFAULT_JUDGE_MODEL = "accounts/fireworks/models/llama4-scout-instruct-basic"
DEFAULT_ORACLE_MODEL = "accounts/fireworks/models/kimi-k2-instruct"
