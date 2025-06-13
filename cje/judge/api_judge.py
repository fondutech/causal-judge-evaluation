"""API-based judge with structured output support."""

from __future__ import annotations
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Type, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .base import BaseJudge, APIJudgeConfig
from .schemas import JudgeScore, JudgeEvaluation, DetailedJudgeEvaluation, JudgeResult
from .providers import (
    StructuredOpenAIProvider,
    StructuredAnthropicProvider,
    StructuredGoogleProvider,
    StructuredProviderStrategy,
    StructuredFireworksProvider,
    StructuredTogetherProvider,
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


class APIJudge(BaseJudge):
    """API-based judge with structured output support."""

    def __init__(self, config: APIJudgeConfig):
        super().__init__(config)
        self.config: APIJudgeConfig = config
        self.provider_strategy = self._get_provider_strategy()
        self.structured_model = self._setup_structured_model()
        self.prompt_template = self._setup_prompt_template()

    def _get_provider_strategy(self) -> StructuredProviderStrategy:
        """Get the appropriate provider strategy based on configuration."""
        strategies: Dict[str, Type[StructuredProviderStrategy]] = {
            "openai": StructuredOpenAIProvider,
            "anthropic": StructuredAnthropicProvider,
            "google": StructuredGoogleProvider,
            "fireworks": StructuredFireworksProvider,
            "together": StructuredTogetherProvider,
        }

        if self.config.provider not in strategies:
            available = list(strategies.keys())
            raise ValueError(
                f"Unsupported provider: {self.config.provider}. "
                f"Available providers: {available}"
            )

        strategy_class = strategies[self.config.provider]
        return strategy_class(
            api_key=self.config.api_key, base_url=self.config.base_url
        )

    def _setup_structured_model(self) -> Runnable[Any, Any]:
        """Set up the model with structured output."""
        # Get the schema class
        schema = self.provider_strategy.get_schema_class(
            self.config.structured_output_schema
        )

        # Get the structured model
        return self.provider_strategy.get_structured_model(
            model_name=self.config.model_name,
            schema=schema,
            method=self.config.structured_output_method,
            temperature=self.config.temperature,
        )

    def _setup_prompt_template(self) -> ChatPromptTemplate:
        """Set up the prompt template for evaluation."""
        # Get the template string
        if self.config.custom_template:
            template_str = self.config.custom_template
        else:
            templates = self._get_templates()
            template_str = templates.get(
                self.config.template,
                templates.get("quick_judge", list(templates.values())[0]),
            )

        # Create prompt template
        return ChatPromptTemplate.from_messages(
            [
                ("system", template_str),
                ("human", "Context: {context}\n\nResponse to evaluate: {response}"),
            ]
        )

    async def _score_async(self, context: str, response: str) -> float:
        """Async scoring with structured output and thinking block parsing."""
        # Create the evaluation chain
        chain = self.prompt_template | self.structured_model

        for attempt in range(self.config.max_retries):
            try:
                # Invoke the chain
                result = await chain.ainvoke(
                    {
                        "context": context,
                        "response": response,
                        **self.config.template_variables,
                    }
                )

                # Handle the include_raw=True response format
                if isinstance(result, dict) and "parsed" in result:
                    if result.get("parsing_error"):
                        # Try to parse thinking blocks from raw output
                        raw_output = result.get("raw", "")
                        parsed_blocks = parse_thinking_blocks(raw_output)

                        if parsed_blocks["thinking"]:
                            # Log the extracted thinking for debugging
                            logger.debug(
                                f"Extracted thinking block: {parsed_blocks['thinking'][:200]}..."
                            )
                            logger.debug(
                                f"Cleaned content: {parsed_blocks['content'][:200]}..."
                            )

                            # Try to parse the cleaned content as JSON
                            try:
                                import json
                                from pydantic import ValidationError

                                # Get the schema class for manual parsing
                                schema_class = self.provider_strategy.get_schema_class(
                                    self.config.structured_output_schema
                                )

                                # Try to parse cleaned content as JSON
                                cleaned_json = json.loads(parsed_blocks["content"])
                                evaluation = schema_class(**cleaned_json)
                                logger.info(
                                    "Successfully parsed response after cleaning thinking blocks"
                                )

                                # Return the normalized score
                                if hasattr(evaluation, "score"):
                                    return float(getattr(evaluation, "score"))
                                elif hasattr(evaluation, "overall_score"):
                                    return float(getattr(evaluation, "overall_score"))
                                else:
                                    raise ValueError(
                                        f"Unexpected evaluation type: {type(evaluation)}"
                                    )

                            except (
                                json.JSONDecodeError,
                                ValidationError,
                                KeyError,
                            ) as parse_error:
                                logger.debug(
                                    f"Cleaned content still failed to parse: {parse_error}"
                                )

                        # Original error handling if thinking block parsing didn't work
                        logger.error(f"Parsing error: {result['parsing_error']}")
                        logger.debug(f"Raw output: {raw_output}")
                        raise ValueError(
                            f"Failed to parse response: {result['parsing_error']}"
                        )
                    evaluation = result["parsed"]
                else:
                    evaluation = result

                # Return the normalized score (0-1 range)
                if hasattr(evaluation, "score"):
                    return float(getattr(evaluation, "score"))
                elif hasattr(evaluation, "overall_score"):
                    return float(getattr(evaluation, "overall_score"))
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

    # ---- NEW: helper to run async coroutines even if an event loop is already running ----
    def _run_sync(self, coro: "Any") -> Any:
        """Execute *coro* and return its result in sync contexts.

        If a loop is already running (e.g. inside Jupyter), we spin up a new
        event loop in a separate context to avoid `RuntimeError: Cannot run the
        event loop while another loop is running`.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # Nested event-loop situation – use an auxiliary loop.
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)

                    # --------------------------------------------------
                    # Graceful shutdown: flush callbacks & asyncgens
                    # --------------------------------------------------
                    # 1) allow "call_soon" callbacks (used by httpx) to run
                    new_loop.run_until_complete(asyncio.sleep(0))

                    # 2) shutdown async generators created by libraries
                    try:
                        new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                    except (AttributeError, RuntimeError):
                        # `shutdown_asyncgens` is Py3.10+; ignore if missing or loop already closed
                        pass

                    # 3) gather any remaining pending tasks (should be none, but just in case)
                    pending = [t for t in asyncio.all_tasks(new_loop) if not t.done()]
                    if pending:
                        for task in pending:
                            task.cancel()
                        try:
                            new_loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        except Exception:
                            pass  # Swallow cleanup exceptions

                    return result
                finally:
                    try:
                        new_loop.close()
                    except Exception:
                        pass  # Ignore cleanup errors
                    finally:
                        asyncio.set_event_loop(None)
        except RuntimeError:
            # No running loop in this thread.
            pass

        # Safe to run directly with proper cleanup
        return asyncio.run(coro)

    # -----------------------------------------------------------------------------
    def score(self, context: str, response: str) -> float:
        """Score a single context-response pair (synchronous helper)."""
        return float(self._run_sync(self._score_async(context, response)))

    async def _score_batch_async(self, samples: List[Dict[str, str]]) -> List[float]:
        """Async batch scoring with structured output."""
        tasks = [
            self._score_async(sample["context"], sample["response"])
            for sample in samples
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[float]:
        """Score a batch of context-response pairs (synchronous wrapper)."""
        return list(self._run_sync(self._score_batch_async(samples)))

    def get_detailed_evaluation(
        self, context: str, response: str
    ) -> Union[JudgeScore, JudgeEvaluation, DetailedJudgeEvaluation]:
        """Get the full structured evaluation (not just the score) with thinking block support."""
        # Create the evaluation chain
        chain = self.prompt_template | self.structured_model

        # Invoke synchronously
        result = chain.invoke(
            {"context": context, "response": response, **self.config.template_variables}
        )

        # Handle the include_raw=True response format
        if isinstance(result, dict) and "parsed" in result:
            if result.get("parsing_error"):
                # Try to parse thinking blocks from raw output
                raw_output = result.get("raw", "")
                parsed_blocks = parse_thinking_blocks(raw_output)

                if parsed_blocks["thinking"]:
                    # Log the extracted thinking for debugging
                    logger.debug(
                        f"Extracted thinking block in detailed eval: {parsed_blocks['thinking'][:200]}..."
                    )
                    logger.debug(
                        f"Cleaned content in detailed eval: {parsed_blocks['content'][:200]}..."
                    )

                    # Try to parse the cleaned content as JSON
                    try:
                        import json
                        from pydantic import ValidationError

                        # Get the schema class for manual parsing
                        schema_class = self.provider_strategy.get_schema_class(
                            self.config.structured_output_schema
                        )

                        # Try to parse cleaned content as JSON
                        cleaned_json = json.loads(parsed_blocks["content"])
                        evaluation = schema_class(**cleaned_json)
                        logger.info(
                            "Successfully parsed detailed evaluation after cleaning thinking blocks"
                        )

                        return evaluation  # type: ignore[return-value, no-any-return]

                    except (
                        json.JSONDecodeError,
                        ValidationError,
                        KeyError,
                    ) as parse_error:
                        logger.debug(
                            f"Cleaned content still failed to parse in detailed eval: {parse_error}"
                        )

                # Original error handling if thinking block parsing didn't work
                raise ValueError(f"Failed to parse response: {result['parsing_error']}")
            return result["parsed"]  # type: ignore[return-value, no-any-return]
        else:
            return result  # type: ignore[return-value, no-any-return]
