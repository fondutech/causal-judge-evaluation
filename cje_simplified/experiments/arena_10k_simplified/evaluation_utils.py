"""
Shared evaluation utilities for judges and oracles.

This module contains reusable implementations for scoring responses
using LLM-based judges and oracles with structured outputs.
"""

import os
import json
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
    """Response containing evaluation score for an AI response."""

    score: float = Field(
        description="Quality score from 0 to 100, where 0 is completely unhelpful and 100 is perfectly helpful",
        ge=0.0,
        le=100.0,
    )


# Evaluator class
class FireworksEvaluator:
    """Fireworks-based evaluator for scoring AI responses."""

    def __init__(
        self,
        model: str,
        system_prompt: str = "You are an AI evaluator. Rate responses from 0 to 100. Always provide a score, even if the response is incomplete or truncated.",
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
        return """<task>
Evaluate the quality of the AI assistant's response to the user's question.
Score from 0-100 based on relevance, helpfulness, clarity, and completeness.
If the response is truncated, score based on what is provided.
</task>

<exchange>
<question>{prompt}</question>
<answer>{response}</answer>
</exchange>

<instruction>
Provide your evaluation score (0-100):
</instruction>"""

    def score(self, prompt: str, response: str) -> JudgeScore:
        """Score a single response."""
        # Format user message from template
        user_message = self.user_prompt_template.format(
            prompt=prompt, response=response
        )

        # Try up to 3 times if structured output fails
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Get structured response
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ]

                result = self.structured_llm.invoke(messages)

                # Check if result is None or invalid
                if result is None:
                    # Log what we sent for debugging
                    print(
                        f"DEBUG: Structured output returned None for prompt: {prompt[:50]}..."
                    )
                    print(f"DEBUG: Response: {response[:50]}...")
                    raise ValueError("Structured output returned None")

                if not isinstance(result, EvaluationResponse):
                    raise ValueError(f"Unexpected result type: {type(result)}")

                # Normalize score from 0-100 to 0-1
                normalized_score = result.score / 100.0

                return JudgeScore(
                    score=normalized_score,
                    metadata={
                        "model": self.model,
                        "raw_score": result.score,  # Keep raw 0-100 score
                        "attempts": attempt + 1,
                    },
                )
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait briefly before retry
                    import time

                    time.sleep(0.5)
                    continue

        # All retries failed
        raise RuntimeError(
            f"Failed to score response with {self.model} after {max_retries} attempts: {str(last_error)}"
        ) from last_error

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
DEFAULT_JUDGE_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
DEFAULT_ORACLE_MODEL = "accounts/fireworks/models/kimi-k2-instruct"
