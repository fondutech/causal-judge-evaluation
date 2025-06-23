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
    JudgeScoreWithCI,
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
        logger.debug(f"Input is not string, converting {type(text).__name__} to string")
        return {"thinking": "", "content": str(text)}

    # Common thinking block patterns with names for debugging
    patterns = [
        # <think>...</think>
        ("think_tags", r"<think>(.*?)</think>"),
        # <thinking>...</thinking>
        ("thinking_tags", r"<thinking>(.*?)</thinking>"),
        # <reason>...</reason>
        ("reason_tags", r"<reason>(.*?)</reason>"),
        # <reasoning>...</reasoning>
        ("reasoning_tags", r"<reasoning>(.*?)</reasoning>"),
        # <analysis>...</analysis>
        ("analysis_tags", r"<analysis>(.*?)</analysis>"),
        # <!-- thinking: ... -->
        ("html_comment", r"<!--\s*thinking:\s*(.*?)\s*-->"),
        # [thinking] ... [/thinking]
        ("bracket_tags", r"\[thinking\](.*?)\[/thinking\]"),
        # <|thinking|>...<|/thinking|> (some models use this)
        ("pipe_tags", r"<\|thinking\|>(.*?)<\|/thinking\|>"),
    ]

    thinking_content = []
    cleaned_text = text
    patterns_found = []

    # Extract all thinking blocks
    for pattern_name, pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            patterns_found.append(pattern_name)
            for match in matches:
                thinking_content.append(match.strip())

            # Remove the thinking blocks from the text
            cleaned_text = re.sub(
                pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
            )

    # Clean up extra whitespace
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text.strip())

    # Log what we found for debugging
    if thinking_content:
        logger.debug(
            f"Found thinking blocks using patterns: {patterns_found}. "
            f"Extracted {len(thinking_content)} blocks totaling {sum(len(b) for b in thinking_content)} chars."
        )
    elif len(text) > 100:  # Only log if there was substantial text
        logger.debug(
            f"No thinking blocks found in {len(text)} chars of text. "
            f"Text preview: {text[:100]}..."
        )

    return {"thinking": "\n\n".join(thinking_content), "content": cleaned_text}


class APIJudge(Judge):
    """Unified API-based judge that returns JudgeScore with uncertainty.

    All scores include uncertainty estimates (mean and variance).
    """

    def __init__(self, config: APIJudgeConfig):
        self.config: APIJudgeConfig = config
        self.provider_strategy = self._get_provider_strategy()
        if self.config.use_structured_output:
            self.structured_model = self._setup_structured_model()
        else:
            # Use regular text model for CI parsing
            self.structured_model = self._setup_text_model()
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
            "JudgeScoreWithCI": JudgeScoreWithCI,
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

    def _setup_text_model(self) -> Runnable:
        """Set up a regular text model (no structured output)."""
        return self.provider_strategy.get_model(  # type: ignore[no-any-return]
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _setup_prompt_template(self) -> ChatPromptTemplate:
        """Set up the prompt template using proper Jinja2 rendering."""
        from ..prompts import UNIFIED_TEMPLATES
        from ..prompts.judge_templates import JUDGE_TEMPLATES
        from ..prompts.template_engine import prepare_judge_template

        # Try judge templates first, then unified templates
        template_info = JUDGE_TEMPLATES.get(self.config.template)
        if not template_info:
            template_info = UNIFIED_TEMPLATES.get(self.config.template)
        if not template_info:
            raise ValueError(f"Unknown template: {self.config.template}")

        template_str = template_info["template"]

        # Merge template default variables with config overrides
        template_vars = template_info.get("variables", {}).copy()
        template_vars.update(self.config.template_variables)

        # Use the proper template engine to prepare for LangChain
        # This will substitute static variables and convert to LangChain format
        langchain_template = prepare_judge_template(template_str, template_vars)

        # Create chat prompt template
        return ChatPromptTemplate.from_template(langchain_template)

    def score(self, context: str, response: str) -> JudgeScore:
        """Score a single context-response pair with uncertainty."""
        result = self._run_sync(self._score_async(context, response))
        return result  # type: ignore[no-any-return]

    async def _score_async(self, context: str, response: str) -> JudgeScore:
        """Async implementation of scoring."""
        # Validate inputs
        if not context or not str(context).strip():
            logger.warning("Empty context provided to judge")
            return JudgeScore(mean=0.0, variance=0.0)

        if not response or not str(response).strip():
            logger.warning("Empty response provided to judge")
            return JudgeScore(mean=0.0, variance=0.0)

        for attempt in range(self.config.max_retries):
            try:
                # Prepare the input variables
                input_vars = {
                    "context": str(context).strip(),
                    "response": str(response).strip(),
                }

                # Format the prompt to see exact text sent to API
                formatted_prompt = self.prompt_template.format_prompt(**input_vars)
                prompt_text = formatted_prompt.to_string()

                # Log the exact prompt being sent
                logger.info(
                    f"[APIJudge] Sending prompt to {self.config.provider}/{self.config.model_name}:"
                )
                logger.info(f"[APIJudge] Template: {self.config.template}")
                logger.info(f"[APIJudge] Prompt length: {len(prompt_text)} chars")
                logger.debug(
                    f"[APIJudge] Full prompt:\n{'-'*80}\n{prompt_text}\n{'-'*80}"
                )

                # Create the chain
                chain = self.prompt_template | self.structured_model

                # Invoke with context and response
                result = await chain.ainvoke(input_vars)

                # Log the raw result from API
                logger.debug(f"[APIJudge] Raw result type: {type(result)}")
                if isinstance(result, dict):
                    logger.debug(f"[APIJudge] Raw result keys: {list(result.keys())}")
                    if "raw" in result:
                        logger.debug(
                            f"[APIJudge] Raw content type: {type(result['raw'])}"
                        )
                        # Handle AIMessage objects
                        if hasattr(result["raw"], "content"):
                            raw_content = result["raw"].content
                            logger.debug(
                                f"[APIJudge] Raw content: {raw_content[:200] if raw_content else '(empty)'}..."
                            )
                    if "parsed" in result:
                        logger.debug(
                            f"[APIJudge] Parsed type: {type(result['parsed'])}"
                        )
                        logger.debug(f"[APIJudge] Parsed content: {result['parsed']}")

                # Handle different result types
                if isinstance(result, dict):
                    # Check for parsing errors
                    if result.get("parsing_error"):
                        # Try to extract from thinking blocks
                        raw_message = result.get("raw", "")
                        # Extract content from AIMessage if needed
                        if hasattr(raw_message, "content"):
                            raw_output = raw_message.content or ""
                        else:
                            raw_output = str(raw_message)
                        parsed_blocks = parse_thinking_blocks(raw_output)

                        logger.debug(
                            f"Parsing error encountered. Attempting recovery from thinking blocks.\n"
                            f"Original error: {result['parsing_error']}\n"
                            f"Raw output length: {len(raw_output)} chars\n"
                            f"Thinking content length: {len(parsed_blocks['thinking'])} chars\n"
                            f"Cleaned content length: {len(parsed_blocks['content'])} chars"
                        )

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
                                        logger.info(
                                            f"Successfully recovered JudgeScore from thinking blocks: "
                                            f"mean={data['mean']}, variance={data['variance']}"
                                        )
                                        return JudgeScore(**data)
                                    elif "score" in data:
                                        # Simple score field
                                        logger.info(
                                            f"Successfully recovered score from thinking blocks: {data['score']}"
                                        )
                                        return JudgeScore(
                                            mean=float(data["score"]), variance=0.0
                                        )
                                    else:
                                        logger.warning(
                                            f"JSON found but missing required fields. "
                                            f"Found keys: {list(data.keys())}"
                                        )
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Failed to parse JSON from cleaned content. "
                                    f"JSON error: {e}\n"
                                    f"Attempted to parse: {json_str[:100]}..."
                                    if "json_str" in locals()
                                    else f"Content: {content[:100]}..."
                                )
                            except ValidationError as e:
                                logger.warning(
                                    f"Parsed JSON but JudgeScore validation failed: {e}\n"
                                    f"Data: {data}"
                                    if "data" in locals()
                                    else "Data: (not parsed)"
                                )
                            except KeyError as e:
                                logger.warning(f"Missing required key: {e}")
                            except Exception as e:
                                logger.warning(
                                    f"Unexpected error during parsing recovery: "
                                    f"{type(e).__name__}: {e}"
                                )

                        # Raise detailed error
                        raise ValueError(
                            f"Failed to parse judge response.\n"
                            f"Original parsing error: {result['parsing_error']}\n"
                            f"Raw output preview: {raw_output[:200]}...\n"
                            f"Cleaned content preview: {parsed_blocks['content'][:200] if parsed_blocks['content'] else '(empty)'}...\n"
                            f"Tip: Check that the model is returning valid JSON with 'mean' and 'variance' fields."
                        )

                    # Extract parsed result
                    evaluation = result.get("parsed", result)
                else:
                    evaluation = result

                # Convert to JudgeScore
                if isinstance(evaluation, JudgeScore):
                    # The JudgeScore schema already handles normalization in its validator
                    logger.info(
                        f"[APIJudge] Final score: mean={evaluation.mean}, variance={evaluation.variance}"
                    )
                    return evaluation
                elif isinstance(evaluation, (JudgeEvaluation, DetailedJudgeEvaluation)):
                    # These already inherit from JudgeScore
                    score = JudgeScore(
                        mean=evaluation.mean, variance=evaluation.variance
                    )
                    logger.info(
                        f"[APIJudge] Final score: mean={score.mean}, variance={score.variance}"
                    )
                    return score
                elif hasattr(evaluation, "score"):
                    # Simple score attribute
                    score = JudgeScore(mean=float(evaluation.score), variance=0.0)
                    logger.info(
                        f"[APIJudge] Final score: mean={score.mean}, variance={score.variance}"
                    )
                    return score
                elif isinstance(evaluation, dict):
                    # Try to construct from dict
                    if "mean" in evaluation:
                        score = JudgeScore(**evaluation)
                        logger.info(
                            f"[APIJudge] Final score: mean={score.mean}, variance={score.variance}"
                        )
                        return score
                    elif "score" in evaluation:
                        score = JudgeScore(
                            mean=float(evaluation["score"]), variance=0.0
                        )
                        logger.info(
                            f"[APIJudge] Final score: mean={score.mean}, variance={score.variance}"
                        )
                        return score
                else:
                    raise ValueError(f"Unexpected evaluation type: {type(evaluation)}")

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.config.max_retries} failed"

                # Add context-specific error information
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    error_msg += " (Rate limit exceeded)"
                elif "timeout" in str(e).lower():
                    error_msg += " (Request timeout)"
                elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
                    error_msg += " (Authentication error - check API key)"
                elif "json" in str(e).lower() or "parsing" in str(e).lower():
                    error_msg += " (Response parsing error)"

                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"{error_msg}: {type(e).__name__}: {e}\n"
                        f"Context length: {len(context)} chars\n"
                        f"Response length: {len(response)} chars"
                    )
                    raise

                logger.warning(f"{error_msg}, retrying in {2**attempt}s: {e}")
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
        import asyncio
        import concurrent.futures
        import threading

        try:
            # Check if we're in an event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, we can use asyncio.run safely
            return asyncio.run(coro)

        # We're in an event loop, need to run in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


class DeterministicAPIJudge(APIJudge):
    """API judge that always returns zero variance.

    For models/prompts that don't estimate uncertainty.
    """

    def __init__(self, config: APIJudgeConfig):
        # Force JudgeScore schema for simple scoring
        config.structured_output_schema = "JudgeScore"
        super().__init__(config)

    async def _score_async(self, context: str, response: str) -> JudgeScore:
        """Score with zero variance."""
        score = await super()._score_async(context, response)
        # Force variance to zero but keep the mean unchanged
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
