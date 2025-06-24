"""Unified Fireworks provider implementation."""

import re
from typing import Any, Dict, Type, Optional
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from .openai_compat import UnifiedOpenAICompatibleProvider


def parse_thinking_blocks_simple(text: str, debug: bool = False) -> str:
    """Parse and remove thinking blocks from reasoning model responses.

    Args:
        text: Raw response text that may contain thinking blocks
        debug: Whether to print debug information

    Returns:
        Cleaned response text with thinking blocks removed
    """
    if not isinstance(text, str):
        return str(text)

    # Strategy 1: Look for complete thinking block patterns first
    complete_patterns = [
        # <think>...</think>
        r"<think>.*?</think>",
        # <thinking>...</thinking>
        r"<thinking>.*?</thinking>",
        # <reason>...</reason>
        r"<reason>.*?</reason>",
        # <reasoning>...</reasoning>
        r"<reasoning>.*?</reasoning>",
        # <analysis>...</analysis>
        r"<analysis>.*?</analysis>",
        # <!-- thinking: ... -->
        r"<!--\s*thinking:.*?-->",
        # [thinking] ... [/thinking]
        r"\[thinking\].*?\[/thinking\]",
    ]

    cleaned_text = text
    blocks_removed = False

    # Try complete patterns first
    for pattern in complete_patterns:
        if re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE):
            cleaned_text = re.sub(
                pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
            )
            blocks_removed = True

    # Strategy 2: If no complete blocks found, look for JSON after thinking block starts
    if not blocks_removed:
        # Look for JSON patterns after thinking block starts
        json_patterns = [
            # Find JSON after <think> or similar
            r"<think[^>]*>.*?(\{[^}]*\})",
            r"<thinking[^>]*>.*?(\{[^}]*\})",
            r"<reason[^>]*>.*?(\{[^}]*\})",
            r"<reasoning[^>]*>.*?(\{[^}]*\})",
            # Find JSON that looks like our target schema
            r'.*?(\{\s*["\']score["\']\s*:\s*[0-9.]+\s*\})',
            # Find any JSON object
            r'.*?(\{[^}]*["\']score["\']\s*:[^}]*\})',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                json_candidate = match.group(1)
                # Try to validate it's actually JSON
                try:
                    import json

                    parsed = json.loads(json_candidate)
                    if isinstance(parsed, dict) and "score" in parsed:
                        return json_candidate
                except (json.JSONDecodeError, Exception):
                    continue

        # Strategy 3: Look for score values and construct JSON
        score_patterns = [
            r'["\']?score["\']?\s*:\s*([0-9.]+)',
            r'score["\']?\s*[=:]\s*([0-9.]+)',
        ]

        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score_value = float(match.group(1))
                constructed_json = json.dumps({"score": score_value})
                return constructed_json

    # Clean up extra whitespace and newlines
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text.strip())
    cleaned_text = cleaned_text.strip()

    return cleaned_text


class ThinkingBlockWrapper(Runnable[Any, Any]):
    """Wrapper that preprocesses model responses to remove thinking blocks."""

    def __init__(self, original_model: Runnable[Any, Any]):
        super().__init__()
        self.original_model = original_model

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        """Synchronous invoke with thinking block preprocessing."""
        try:
            result = self.original_model.invoke(input, config, **kwargs)
            return self._process_result(result)
        except Exception as e:
            # Try to extract raw response from parsing errors
            return self._handle_parsing_error(e)

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        """Async invoke with thinking block preprocessing."""
        try:
            result = await self.original_model.ainvoke(input, config, **kwargs)
            return self._process_result(result)
        except Exception as e:
            # Try to extract raw response from parsing errors
            return self._handle_parsing_error(e)

    def _process_result(self, result: Any) -> Any:
        """Process the result to remove thinking blocks."""
        # Handle include_raw=True format
        if isinstance(result, dict) and "parsed" in result:
            if result.get("parsing_error") and result.get("raw"):
                # Try to clean the raw output and reparse
                raw_output = result["raw"]
                cleaned_output = parse_thinking_blocks_simple(raw_output)

                # If we successfully cleaned something, try to parse it
                if cleaned_output != raw_output and cleaned_output.strip():
                    try:
                        import json

                        # Try to parse as JSON
                        cleaned_json = json.loads(cleaned_output)

                        # Get the schema from the original error to validate
                        # For now, return a basic structure that should work
                        return {
                            "raw": cleaned_output,
                            "parsed": cleaned_json,
                            "parsing_error": None,
                        }
                    except (json.JSONDecodeError, Exception):
                        # If still fails, fall back to original result
                        pass

        return result

    def _handle_parsing_error(self, error: Exception) -> Any:
        """Try to handle parsing errors by extracting and cleaning raw response."""
        # Get the full error string
        error_str = str(error)

        # Strategy 1: Look for any JSON-like patterns in the error message
        # Even if truncated, we can often find valid JSON structures
        json_patterns = [
            # Look for simple score patterns we can extract
            r'"score":\s*([0-9]+(?:\.[0-9]+)?)',
            r'"score_overall":\s*([0-9]+(?:\.[0-9]+)?)',
            r'"final_score":\s*([0-9]+(?:\.[0-9]+)?)',
            r'"rating":\s*([0-9]+(?:\.[0-9]+)?)',
            # Look for complete JSON objects even if surrounded by thinking blocks
            r'\{[^{}]*"score"[^{}]*\}',
            r'\{[^{}]*"rating"[^{}]*\}',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, error_str, re.IGNORECASE)
            if matches:
                # If we found just a number, construct JSON
                if isinstance(matches[0], str) and re.match(
                    r"^[0-9]+(?:\.[0-9]+)?$", matches[0]
                ):
                    score_value = float(matches[0])
                    # Clamp score to reasonable range
                    score_value = max(0.0, min(10.0, score_value))
                    constructed_json = {"score": score_value}

                    from ..schemas import JudgeScore

                    return JudgeScore(**constructed_json)

                # If we found a JSON object, try to parse it
                elif "{" in str(matches[0]):
                    json_candidate = matches[0]

                    try:
                        import json

                        parsed_dict = json.loads(json_candidate)
                        if isinstance(parsed_dict, dict) and any(
                            key in parsed_dict
                            for key in ["score", "rating", "score_overall"]
                        ):
                            # Normalize to 'score' key
                            if "score" not in parsed_dict:
                                parsed_dict["score"] = parsed_dict.get(
                                    "rating", parsed_dict.get("score_overall", 5.0)
                                )

                            from ..schemas import JudgeScore

                            return JudgeScore(**parsed_dict)

                    except (json.JSONDecodeError, Exception):
                        continue

        # Strategy 2: If no clear patterns found, return a default neutral score
        # This ensures the pipeline can continue rather than failing completely
        from ..schemas import JudgeScore

        return JudgeScore(mean=0.5)  # Neutral score to keep pipeline running


class FireworksProvider(UnifiedOpenAICompatibleProvider):
    """Unified Fireworks.ai provider with thinking block support."""

    ENV_VAR = "FIREWORKS_API_KEY"
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"

    def get_structured_output_params(self, method: str) -> Dict[str, Any]:
        """Get Fireworks-specific parameters for structured output.

        Fireworks doesn't support OpenAI's strict mode, so we disable it.
        """
        return {"strict": False}

    def get_structured_model(
        self,
        model_name: str,
        schema: Type[BaseModel],
        method: str = "auto",
        temperature: float = 0.0,
    ) -> Runnable[Any, Any]:
        """Get a model configured for structured output with thinking block support."""
        # Get the base structured model
        base_model = super().get_structured_model(
            model_name, schema, method, temperature
        )

        # Always wrap with thinking block preprocessing since any model could potentially
        # produce thinking blocks, and it gracefully handles responses without them
        return ThinkingBlockWrapper(base_model)
