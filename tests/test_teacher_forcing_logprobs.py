"""Test teacher forcing log probability extraction.

This module tests the critical bug fix where prompt tokens were being included
in response log probabilities, causing unreasonably low values for short responses.
"""

import pytest
from cje.loggers.api_policy import sum_response_logprobs_tail


def test_sum_response_logprobs_tail() -> None:
    """Test the sum_response_logprobs_tail utility function."""
    # Test normal case
    all_logprobs = [-1.0, -2.0, -3.0, -4.0, -5.0]
    result = sum_response_logprobs_tail(all_logprobs, response_token_count=2)
    assert result == -9.0  # -4.0 + -5.0

    # Test edge case: more tokens requested than available
    result = sum_response_logprobs_tail(all_logprobs, response_token_count=10)
    assert result == sum(all_logprobs)

    # Test edge case: zero tokens
    result = sum_response_logprobs_tail(all_logprobs, response_token_count=0)
    assert result == 0.0

    # Test with positive logprobs (shouldn't happen but should handle)
    mixed_logprobs = [-1.0, 0.5, -2.0]
    result = sum_response_logprobs_tail(mixed_logprobs, response_token_count=2)
    assert result == -1.5  # 0.5 + (-2.0)

    # Test with None values (should filter them out)
    logprobs_with_none = [-1.0, None, -2.0, -3.0]
    result = sum_response_logprobs_tail(logprobs_with_none, response_token_count=2)
    # Should take last 2 non-None values: -2.0 + -3.0
    assert result == -5.0


class TestLogprobReasonableness:
    """Document expected log probability ranges for common responses.

    The bug would cause single-word responses to have log probabilities < -20
    due to including prompt tokens. Fixed version should have much more
    reasonable values.
    """

    @pytest.mark.parametrize(
        "response,expected_reasonable_range",
        [
            ("Yes", (-5, -0.1)),  # Single word, very common
            ("Cabbages", (-10, -1)),  # Single word, less common
            ("No", (-5, -0.1)),  # Very common response
            ("I don't know", (-15, -2)),  # Multi-word common phrase
        ],
    )
    def test_reasonable_logprob_ranges(
        self, response: str, expected_reasonable_range: tuple[float, float]
    ) -> None:
        """Document reasonable log probability ranges for responses.

        This test serves as documentation of expected behavior.
        Full integration testing would be needed to verify actual API behavior.
        """
        # The bug manifested as:
        # - "Cabbages" having logprob of -21.625 (way too low)
        # - This suggested prompt tokens were included
        #
        # After fix:
        # - Single words should typically be in range -10 to -0.1
        # - Common words like "Yes"/"No" should be > -5
        # - Less common words like "Cabbages" might be -10 to -1
        pass


# Integration test sketch (would require real API calls):
#
# def test_teacher_forcing_integration():
#     """Test that teacher forcing extracts only response logprobs."""
#     runner = APIPolicyRunner(provider="fireworks", model_name="llama-3.1-8b", temperature=0.5)
#
#     # Generate a simple response
#     prompt = "What vegetable is green?"
#     response = "Cabbage"
#
#     # Get logprob using teacher forcing
#     logprob = runner._teacher_forcing_logprob(prompt, response)
#
#     # Should be reasonable for a single word
#     assert -10 < logprob < -0.1, f"Logprob {logprob} suggests prompt tokens included"
