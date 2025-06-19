"""Uncertainty-aware judge interface.

This module provides judges that always return uncertainty estimates,
making uncertainty a first-class citizen in the evaluation pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, cast, Type
from dataclasses import dataclass, field
import asyncio
import logging

from .schemas import JudgeScore
from ..judge.base import APIJudgeConfig
from ..judge.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    FireworksProvider,
    TogetherProvider,
)

logger = logging.getLogger(__name__)


class UncertaintyAwareJudge(ABC):
    """Base class for judges that always return uncertainty estimates."""

    @abstractmethod
    def score(self, context: str, response: str) -> JudgeScore:
        """Score a single context-response pair with uncertainty.

        Args:
            context: Context/question
            response: Model response

        Returns:
            JudgeScore with mean and variance
        """
        pass

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Score multiple samples with uncertainty.

        Default implementation calls score() for each sample.
        Can be overridden for more efficient batch processing.

        Args:
            samples: List of dicts with 'context' and 'response' keys
            disable_progress: Whether to disable progress bar

        Returns:
            List of JudgeScore objects
        """
        from ..utils.progress import track

        return [
            self.score(s["context"], s["response"])
            for s in track(
                samples,
                description="Scoring with uncertainty",
                disable=disable_progress,
            )
        ]


@dataclass
class UncertaintyJudgeConfig(APIJudgeConfig):
    """Configuration for uncertainty-aware judges."""

    # Override default schema to always use uncertainty
    structured_output_schema: str = "UncertainJudgeScore"

    # Beta distribution parameters for variance estimation
    beta_concentration: float = 10.0  # Higher = more confident
    use_adaptive_concentration: bool = True  # Adjust based on score proximity to 0.5

    # Uncertainty prompt additions
    include_uncertainty_prompt: bool = True
    uncertainty_instruction: str = (
        "Also indicate your confidence in this score. "
        "Higher confidence means lower variance in your score estimate."
    )

    def __post_init__(self) -> None:
        """Ensure uncertainty schema is always used."""
        # Force uncertainty schema
        self.structured_output_schema = "UncertainJudgeScore"

        # Ensure we're using structured output
        self.use_structured_output = True

        # Add uncertainty to valid schemas if not present
        if hasattr(self, "_VALID_SCHEMAS"):
            self._VALID_SCHEMAS.add("UncertainJudgeScore")

        super().__post_init__()


class UncertaintyAPIJudge(UncertaintyAwareJudge):
    """API-based judge that always returns uncertainty estimates."""

    def __init__(self, config: UncertaintyJudgeConfig):
        self.config = config
        self.provider_strategy = self._get_provider_strategy()
        self.structured_model = self._setup_structured_model()
        self.prompt_template = self._setup_prompt_template()

    def _get_provider_strategy(self) -> Any:
        """Get appropriate provider strategy."""
        strategies: Dict[str, Any] = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
            "fireworks": FireworksProvider,
            "together": TogetherProvider,
        }

        provider_class = strategies.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        # Instantiate provider with proper arguments
        if self.config.base_url:
            return provider_class(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        else:
            return provider_class(api_key=self.config.api_key)

    def _setup_structured_model(self) -> Any:
        """Set up the structured output model."""
        # Use JudgeScore which has uncertainty built-in
        return self.provider_strategy.get_structured_model(
            model_name=self.config.model_name,
            schema=JudgeScore,  # JudgeScore already has mean/variance
            temperature=self.config.temperature,
            method=self.config.structured_output_method,
        )

    def _setup_prompt_template(self) -> Any:
        """Set up the prompt template with uncertainty instructions."""
        from langchain_core.prompts import ChatPromptTemplate
        from ..prompts import UNIFIED_TEMPLATES

        # Get base template
        template_info = UNIFIED_TEMPLATES.get(
            self.config.template, UNIFIED_TEMPLATES.get("quick_judge")
        )
        if template_info is None:
            raise ValueError(f"Template not found: {self.config.template}")
        base_template = template_info["template"]

        # Add uncertainty instruction if requested
        if self.config.include_uncertainty_prompt:
            base_template = base_template.rstrip()
            if not base_template.endswith("."):
                base_template += "."
            base_template += f"\n\n{self.config.uncertainty_instruction}"

        # Create chat prompt template
        return ChatPromptTemplate.from_template(base_template)

    async def _score_async(self, context: str, response: str) -> JudgeScore:
        """Async scoring with uncertainty."""
        chain = self.prompt_template | self.structured_model

        for attempt in range(self.config.max_retries):
            try:
                result = await chain.ainvoke(
                    {
                        "context": context,
                        "response": response,
                        **self.config.template_variables,
                    }
                )

                # Handle structured output response
                if isinstance(result, dict) and "parsed" in result:
                    if result.get("parsing_error"):
                        raise ValueError(f"Parse error: {result['parsing_error']}")
                    result = result["parsed"]

                # Convert to our JudgeScore
                return JudgeScore(
                    mean=float(result.score_mean),
                    variance=float(result.score_variance),
                )

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Failed after {self.config.max_retries} attempts: {e}"
                    )
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2**attempt)

        # This should never be reached due to the raise in the exception handler
        raise RuntimeError("Exhausted retries without success")

    def score(self, context: str, response: str) -> JudgeScore:
        """Score with uncertainty (sync wrapper)."""
        return cast(JudgeScore, self._run_sync(self._score_async(context, response)))

    async def _score_batch_async(
        self, samples: List[Dict[str, str]]
    ) -> List[JudgeScore]:
        """Async batch scoring."""
        tasks = [self._score_async(s["context"], s["response"]) for s in samples]
        return await asyncio.gather(*tasks)

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Batch scoring with uncertainty."""
        return cast(List[JudgeScore], self._run_sync(self._score_batch_async(samples)))

    def _run_sync(self, coro: Any) -> Any:
        """Run async code in sync context."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Handle nested event loop (e.g., Jupyter)
                import nest_asyncio

                nest_asyncio.apply()
        except RuntimeError:
            pass

        return asyncio.run(coro)


class DeterministicJudge(UncertaintyAwareJudge):
    """Wrapper that adds zero variance to deterministic judges.

    This allows using legacy judges in the uncertainty-aware pipeline.
    """

    def __init__(self, base_judge: Any):
        """Wrap a traditional judge to return zero variance.

        Args:
            base_judge: Traditional judge with score() method
        """
        self.base_judge = base_judge

    def score(self, context: str, response: str) -> JudgeScore:
        """Score with zero variance."""
        score = self.base_judge.score(context, response)
        return JudgeScore(mean=float(score), variance=0.0)

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Batch scoring with zero variance."""
        scores = self.base_judge.score_batch(samples, disable_progress)
        return [JudgeScore(mean=float(s), variance=0.0) for s in scores]


class MockUncertaintyJudge(UncertaintyAwareJudge):
    """Mock judge for testing uncertainty propagation."""

    def __init__(
        self,
        base_score: float = 0.7,
        base_variance: float = 0.05,
        noise_std: float = 0.1,
    ):
        self.base_score = base_score
        self.base_variance = base_variance
        self.noise_std = noise_std

    def score(self, context: str, response: str) -> JudgeScore:
        """Generate mock score with variance."""
        import numpy as np

        # Add some noise to base score
        noise = np.random.normal(0, self.noise_std)
        score = np.clip(self.base_score + noise, 0, 1)

        # Variance increases for scores near 0.5 (maximum uncertainty)
        distance_from_middle = abs(score - 0.5)
        variance_multiplier = 1 + 2 * (1 - 2 * distance_from_middle)
        variance = self.base_variance * variance_multiplier

        return JudgeScore(mean=score, variance=variance)
