"""
Validation module for completions templates to ensure teacher forcing works correctly.

This module provides validation checks to ensure that:
1. Templates are correctly formatting prompts
2. Log probabilities are in reasonable ranges
3. The system fails loudly rather than silently when there's a template mismatch
"""

from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationCase:
    """A test case for template validation."""

    messages: List[Dict[str, str]]
    expected_response: str
    min_logprob: float  # Minimum acceptable log probability
    max_logprob: float  # Maximum acceptable log probability (usually 0.0)
    description: str


# Standard validation cases that should work for most models
STANDARD_VALIDATION_CASES = [
    ValidationCase(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        expected_response="4",
        min_logprob=-2.0,  # Single token "4" should have high probability
        max_logprob=0.0,
        description="Simple math answer",
    ),
    ValidationCase(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        expected_response="The answer is 4.",
        min_logprob=-10.0,  # Multi-token response, more variation expected
        max_logprob=0.0,
        description="Full sentence math answer",
    ),
    ValidationCase(
        messages=[{"role": "user", "content": "Name a vegetable."}],
        expected_response="Cabbage",
        min_logprob=-5.0,  # Single word, should be reasonably probable
        max_logprob=0.0,
        description="Single word vegetable",
    ),
    ValidationCase(
        messages=[{"role": "user", "content": "Complete: The sky is"}],
        expected_response="blue",
        min_logprob=-3.0,  # Very common completion
        max_logprob=0.0,
        description="Common completion",
    ),
]


class TemplateValidationError(Exception):
    """Raised when template validation fails."""

    pass


class TemplateMismatchWarning(UserWarning):
    """Warning for potential template mismatches."""

    pass


class TemplateValidator:
    """Validates completions templates for teacher forcing."""

    def __init__(self, api_runner: Any):
        """
        Initialize validator with an API runner.

        Args:
            api_runner: An APIPolicyRunner instance configured for the model/provider
        """
        self.api_runner = api_runner

    def validate_single_case(
        self, case: ValidationCase, verbose: bool = True
    ) -> Tuple[bool, float, str]:
        """
        Validate a single test case.

        Args:
            case: The validation case to test
            verbose: Whether to log detailed information

        Returns:
            Tuple of (success, actual_logprob, error_message)
        """
        try:
            # Compute log probability using teacher forcing
            logprob = self.api_runner.log_prob(
                context=self._messages_to_context(case.messages),
                response=case.expected_response,
            )

            # Check if logprob is in expected range
            if logprob < case.min_logprob:
                # Add diagnostic hints based on how low the logprob is
                if logprob < -20:
                    hint = (
                        "\n\nDIAGNOSTIC: Extremely low log probability suggests:\n"
                        "1. Wrong template format (e.g., using Llama 4 template for Llama 3 model)\n"
                        "2. Template tokens being scored instead of response tokens\n"
                        "3. Provider doesn't support echo=True for this model\n"
                        "\nTry: Checking if model name matches template format"
                    )
                elif logprob < -15:
                    hint = (
                        "\n\nDIAGNOSTIC: Very low log probability suggests:\n"
                        "1. Possible template mismatch\n"
                        "2. Response tokens not being extracted correctly\n"
                        "\nTry: Verifying the template format matches the model"
                    )
                else:
                    hint = ""

                error_msg = (
                    f"{case.description}: Log probability {logprob:.3f} is too low "
                    f"(expected >= {case.min_logprob}). This suggests the template "
                    f"may be incorrectly formatting the prompt.{hint}"
                )
                if verbose:
                    logger.warning(error_msg)
                return False, logprob, error_msg

            if logprob > case.max_logprob:
                error_msg = (
                    f"{case.description}: Log probability {logprob:.3f} is too high "
                    f"(expected <= {case.max_logprob}). This is unusual and may "
                    f"indicate a calculation error."
                )
                if verbose:
                    logger.warning(error_msg)
                return False, logprob, error_msg

            if verbose:
                logger.info(
                    f"✓ {case.description}: Log probability {logprob:.3f} is in "
                    f"expected range [{case.min_logprob}, {case.max_logprob}]"
                )

            return True, logprob, ""

        except Exception as e:
            error_msg = f"{case.description}: Failed with error: {str(e)}"
            if verbose:
                logger.error(error_msg)
            return False, float("-inf"), error_msg

    def validate_all(
        self,
        cases: Optional[List[ValidationCase]] = None,
        fail_fast: bool = True,
        verbose: bool = True,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate multiple test cases.

        Args:
            cases: List of validation cases (defaults to STANDARD_VALIDATION_CASES)
            fail_fast: Whether to stop on first failure
            verbose: Whether to log detailed information

        Returns:
            Tuple of (all_passed, results_list)

        Raises:
            TemplateValidationError: If validation fails and fail_fast=True
        """
        if cases is None:
            cases = STANDARD_VALIDATION_CASES

        results = []
        all_passed = True

        if verbose:
            logger.info(
                f"Validating teacher forcing for {self.api_runner.provider}:"
                f"{self.api_runner.model_name} with {self.api_runner.template.__class__.__name__}"
            )

        for case in cases:
            success, logprob, error_msg = self.validate_single_case(case, verbose)

            result = {
                "description": case.description,
                "success": success,
                "logprob": logprob,
                "expected_range": [case.min_logprob, case.max_logprob],
                "error": error_msg,
            }
            results.append(result)

            if not success:
                all_passed = False
                if fail_fast:
                    raise TemplateValidationError(
                        f"Template validation failed: {error_msg}\n"
                        f"Model: {self.api_runner.model_name}\n"
                        f"Template: {self.api_runner.template.__class__.__name__}\n"
                        f"This likely means the template is not correctly formatting "
                        f"prompts for this model."
                    )

        if verbose:
            if all_passed:
                logger.info("✅ All validation cases passed!")
            else:
                logger.warning("❌ Some validation cases failed")

        return all_passed, results

    def validate_template_format(self, verbose: bool = True) -> bool:
        """
        Validate that the template is producing expected format.

        This is a basic check that the template methods work and produce
        non-empty strings with expected tokens.

        Returns:
            True if format validation passes
        """
        try:
            # Test messages
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            response = "Hi there"

            # Test formatting methods
            with_response = self.api_runner.template.format_with_response(
                messages, response
            )
            without_response = self.api_runner.template.format_without_response(
                messages
            )
            eos_token = self.api_runner.template.get_eos_token()

            # Basic checks
            if not with_response or not without_response:
                raise TemplateValidationError("Template produced empty strings")

            if response not in with_response:
                raise TemplateValidationError(
                    f"Response '{response}' not found in formatted prompt"
                )

            if without_response not in with_response:
                raise TemplateValidationError(
                    "Format without response should be prefix of format with response"
                )

            if eos_token and eos_token not in with_response:
                logger.warning(
                    f"EOS token '{eos_token}' not found in formatted prompt. "
                    f"This may be normal for some templates."
                )

            if verbose:
                logger.info("✓ Template format validation passed")
                logger.debug(f"Sample formatted prompt: {with_response[:100]}...")

            return True

        except Exception as e:
            if verbose:
                logger.error(f"Template format validation failed: {e}")
            raise TemplateValidationError(f"Template format validation failed: {e}")

    def _messages_to_context(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to context string for log_prob method."""
        # This is a simplified version - in practice, this should match
        # how the API runner expects context to be formatted
        context_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                context_parts.append(f"System: {content}")
            elif role == "user":
                context_parts.append(content)
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
        return "\n".join(context_parts)


def validate_teacher_forcing(
    api_runner: Any,
    custom_cases: Optional[List[ValidationCase]] = None,
    fail_fast: bool = True,
    verbose: bool = True,
) -> bool:
    """
    Convenience function to validate teacher forcing for an API runner.

    Args:
        api_runner: An APIPolicyRunner instance
        custom_cases: Optional custom validation cases
        fail_fast: Whether to stop on first failure
        verbose: Whether to log detailed information

    Returns:
        True if all validations pass

    Raises:
        TemplateValidationError: If validation fails and fail_fast=True
    """
    validator = TemplateValidator(api_runner)

    # First validate template format
    validator.validate_template_format(verbose)

    # Then validate with test cases
    all_passed, results = validator.validate_all(
        cases=custom_cases, fail_fast=fail_fast, verbose=verbose
    )

    return all_passed


# Specific validation cases for known problematic scenarios
PROBLEMATIC_CASES = [
    ValidationCase(
        messages=[{"role": "user", "content": "Name a vegetable"}],
        expected_response="Cabbages",  # The famous -21.625 case
        min_logprob=-5.0,
        max_logprob=0.0,
        description="Cabbages case (historically problematic with wrong template)",
    ),
    ValidationCase(
        messages=[{"role": "user", "content": "1+1="}],
        expected_response="2",
        min_logprob=-1.0,  # Single digit should be very high probability
        max_logprob=0.0,
        description="Single digit math",
    ),
]


def diagnose_template_mismatch(logprobs: List[float]) -> str:
    """Diagnose potential template mismatches based on log probability patterns.

    Args:
        logprobs: List of log probabilities from validation cases

    Returns:
        Diagnostic message with recommendations
    """
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else 0

    if avg_logprob < -20:
        return (
            "SEVERE TEMPLATE MISMATCH DETECTED:\n"
            "The extremely low log probabilities (-20 or lower) indicate:\n"
            "\n"
            "1. **Wrong template version**: You may be using a Llama 4 template "
            "with a Llama 3 model or vice versa.\n"
            "   - Llama 3 uses: <|start_header_id|>...<|end_header_id|>\n"
            "   - Llama 4 uses: <|header_start|>...<|header_end|>\n"
            "\n"
            "2. **Provider compatibility**: The provider may not support echo=True "
            "for this specific model.\n"
            "   - Together AI: Only supports echo=True for Llama 3.x models\n"
            "   - Fireworks: Supports all models\n"
            "\n"
            "3. **Token extraction issue**: Template tokens (like <|eot|>) may be "
            "included in the response scoring.\n"
            "\n"
            "RECOMMENDED ACTIONS:\n"
            "- Verify model version matches template format\n"
            "- Check provider documentation for echo=True support\n"
            "- Try a different model version (e.g., Llama 3.3 instead of Llama 4)"
        )
    elif avg_logprob < -15:
        return (
            "POSSIBLE TEMPLATE ISSUE DETECTED:\n"
            "The low log probabilities suggest a potential configuration issue.\n"
            "\n"
            "Consider:\n"
            "- Double-checking the model name and version\n"
            "- Verifying the template format is correct\n"
            "- Testing with a known working model/provider combination"
        )
    else:
        return ""
