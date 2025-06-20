"""Unified API-based judge that returns JudgeScore with uncertainty."""

from __future__ import annotations
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Type, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError
import json

from .base import BaseJudge, APIJudgeConfig
from .judges import Judge, DeterministicJudge, ProbabilisticJudge
from .schemas import (
    JudgeScore,
    JudgeEvaluation,
    DetailedJudgeEvaluation,
    JudgeResult,
)
from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    FireworksProvider,
    TogetherProvider,
)

logger = logging.getLogger(__name__)


def parse_thinking_blocks(text: str) -> Dict[str, str]:
    """Parse and extract thinking blocks from reasoning model responses.

    Args:
        text: Raw response text that may contain thinking blocks

    Returns:
        Dict with 'thinking' (extracted reasoning) and 'content' (cleaned response)
    """
    if not isinstance(text, str):
        return {"thinking": "", "content": str(text)}

    # Common thinking block patterns
    patterns = [
        # <think>...</think>
        r"<think>(.*?)</think>",
        # <thinking>...</thinking>
        r"<thinking>(.*?)</thinking>",
        # <reason>...</reason>
        r"<reason>(.*?)</reason>",
        # <reasoning>...</reasoning>
        r"<reasoning>(.*?)</reasoning>",
        # <analysis>...</analysis>
        r"<analysis>(.*?)</analysis>",
        # <!-- thinking: ... -->
        r"<!--\s*thinking:\s*(.*?)\s*-->",
        # [thinking] ... [/thinking]
        r"\[thinking\](.*?)\[/thinking\]",
    ]

    thinking_content = []
    cleaned_text = text

    # Extract all thinking blocks
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            thinking_content.append(match.strip())

        # Remove the thinking blocks from the text
        cleaned_text = re.sub(
            pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
        )

    # Clean up extra whitespace
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text.strip())

    return {"thinking": "\n\n".join(thinking_content), "content": cleaned_text}


class APIJudge(Judge):
    """Unified API-based judge that returns JudgeScore with uncertainty.

    All scores include uncertainty estimates (mean and variance).
    """

    def __init__(self, config: APIJudgeConfig):
        self.config: APIJudgeConfig = config
        self.provider_strategy = self._get_provider_strategy()
        self.structured_model = self._setup_structured_model()
        self.prompt_template = self._setup_prompt_template()

    def _get_provider_strategy(self) -> Any:
        """Get the appropriate provider strategy based on configuration."""
        strategies: Dict[str, Type[Any]] = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
            "fireworks": FireworksProvider,
            "together": TogetherProvider,
        }

        provider_class = strategies.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        # Instantiate the provider
        if self.config.base_url:
            return provider_class(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        else:
            return provider_class(api_key=self.config.api_key)

    def _setup_structured_model(self) -> Runnable:
        """Set up the structured output model."""
        schema_mapping: Dict[str, Type[BaseModel]] = {
            "JudgeScore": JudgeScore,
            "JudgeEvaluation": JudgeEvaluation,
            "DetailedJudgeEvaluation": DetailedJudgeEvaluation,
        }

        schema_class = schema_mapping.get(self.config.structured_output_schema)
        if not schema_class:
            raise ValueError(f"Unknown schema: {self.config.structured_output_schema}")

        model = self.provider_strategy.get_structured_model(
            model_name=self.config.model_name,
            schema=schema_class,
            temperature=self.config.temperature,
            method=self.config.structured_output_method,
        )
        return model  # type: ignore[no-any-return]

    def _setup_prompt_template(self) -> ChatPromptTemplate:
        """Set up the prompt template."""
        from ..prompts import UNIFIED_TEMPLATES

        # Get template content
        template_info = UNIFIED_TEMPLATES.get(self.config.template)
        if not template_info:
            raise ValueError(f"Unknown template: {self.config.template}")

        template_str = template_info["template"]

        # Create chat prompt template
        return ChatPromptTemplate.from_template(template_str)

    def score(self, context: str, response: str) -> JudgeScore:
        """Score a single context-response pair with uncertainty."""
        result = self._run_sync(self._score_async(context, response))
        return result  # type: ignore[no-any-return]

    async def _score_async(self, context: str, response: str) -> JudgeScore:
        """Async implementation of scoring."""
        for attempt in range(self.config.max_retries):
            try:
                # Create the chain
                chain = self.prompt_template | self.structured_model

                # Invoke with context and response
                result = await chain.ainvoke(
                    {
                        "context": context,
                        "response": response,
                        **self.config.template_variables,
                    }
                )

                # Handle different result types
                if isinstance(result, dict):
                    # Check for parsing errors
                    if result.get("parsing_error"):
                        # Try to extract from thinking blocks
                        raw_output = result.get("raw", "")
                        parsed_blocks = parse_thinking_blocks(raw_output)

                        if parsed_blocks["content"]:
                            try:
                                # Try to parse the cleaned content
                                content = parsed_blocks["content"]
                                # Simple JSON extraction
                                if "{" in content and "}" in content:
                                    json_str = content[
                                        content.find("{") : content.rfind("}") + 1
                                    ]
                                    data = json.loads(json_str)

                                    # Convert to JudgeScore
                                    if "mean" in data and "variance" in data:
                                        return JudgeScore(**data)
                                    elif "score" in data:
                                        # Simple score field
                                        return JudgeScore(
                                            mean=float(data["score"]), variance=0.0
                                        )
                            except (
                                json.JSONDecodeError,
                                ValidationError,
                                KeyError,
                            ) as e:
                                logger.debug(f"Failed to parse cleaned content: {e}")

                        # Raise original error
                        raise ValueError(f"Parsing error: {result['parsing_error']}")

                    # Extract parsed result
                    evaluation = result.get("parsed", result)
                else:
                    evaluation = result

                # Convert to JudgeScore
                if isinstance(evaluation, JudgeScore):
                    return evaluation
                elif isinstance(evaluation, (JudgeEvaluation, DetailedJudgeEvaluation)):
                    # These already inherit from JudgeScore
                    return JudgeScore(
                        mean=evaluation.mean, variance=evaluation.variance
                    )
                elif hasattr(evaluation, "score"):
                    # Simple score attribute
                    return JudgeScore(mean=float(evaluation.score), variance=0.0)
                elif isinstance(evaluation, dict):
                    # Try to construct from dict
                    if "mean" in evaluation:
                        return JudgeScore(**evaluation)
                    elif "score" in evaluation:
                        return JudgeScore(mean=float(evaluation["score"]), variance=0.0)
                else:
                    raise ValueError(f"Unexpected evaluation type: {type(evaluation)}")

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Failed to score after {self.config.max_retries} attempts: {e}"
                    )
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                await asyncio.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError("Unreachable code")

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Score multiple samples concurrently."""
        result = self._run_sync(self._score_batch_async(samples, disable_progress))
        return result  # type: ignore[no-any-return]

    async def _score_batch_async(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Async batch scoring implementation."""
        from ..utils.progress import track

        # Create tasks for all samples
        tasks = []
        for sample in track(
            samples,
            description=f"Scoring with {self.config.model_name}",
            disable=disable_progress,
        ):
            task = self._score_async(sample["context"], sample["response"])
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to score sample {i}: {result}")
                # Return a low score with high uncertainty for failures
                scores.append(JudgeScore(mean=0.0, variance=0.25))
            else:
                assert isinstance(result, JudgeScore)
                scores.append(result)

        return scores

    def _run_sync(self, coro: Any) -> Any:
        """Execute async code in sync context."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Nested event-loop situation – use auxiliary loop
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)

                    # Graceful shutdown
                    new_loop.run_until_complete(asyncio.sleep(0))

                    try:
                        new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                    except (AttributeError, RuntimeError):
                        pass

                    # Cancel remaining tasks
                    pending = [t for t in asyncio.all_tasks(new_loop) if not t.done()]
                    for task in pending:
                        task.cancel()

                    if pending:
                        try:
                            new_loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        except Exception:
                            pass

                    return result
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass
                    finally:
                        asyncio.set_event_loop(None)
        except RuntimeError:
            # No running loop
            pass

        # Run normally
        return asyncio.run(coro)


class DeterministicAPIJudge(APIJudge):
    """API judge that always returns zero variance.

    For models/prompts that don't estimate uncertainty.
    """

    def __init__(self, config: APIJudgeConfig):
        # Force JudgeScore schema for simple scoring
        config.structured_output_schema = "JudgeScore"
        # Ensure mean-only fields
        super().__init__(config)

    async def _score_async(self, context: str, response: str) -> JudgeScore:
        """Score with zero variance."""
        score = await super()._score_async(context, response)
        # Force variance to zero
        return JudgeScore(mean=score.mean, variance=0.0)


class MCAPIJudge(APIJudge, ProbabilisticJudge):
    """API judge that uses Monte Carlo sampling for uncertainty.

    Scores multiple times with temperature > 0 to estimate variance.
    """

    def __init__(self, config: APIJudgeConfig, n_samples: int = 5):
        # Ensure temperature > 0 for sampling
        if config.temperature == 0:
            config.temperature = 0.3
        super().__init__(config)
        self.n_samples = n_samples

    def score(self, context: str, response: str) -> JudgeScore:
        """Score using Monte Carlo sampling."""
        return self.score_with_samples(context, response, self.n_samples)

    def _sample_score(self, context: str, response: str) -> float:
        """Sample a single score."""
        score = self._run_sync(self._score_async(context, response))
        return float(score.mean)
