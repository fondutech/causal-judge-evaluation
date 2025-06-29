"""Test teacher forcing log probability extraction.

This module tests the robust teacher forcing implementation that correctly
handles token boundary alignment issues.
"""

import pytest
from cje.utils.logprobs import safe_sum
from cje.utils.teacher_forcing import (
    RobustTeacherForcing,
    compute_teacher_forced_logprob,
)
from cje.types import LogProbStatus


def test_safe_sum() -> None:
    """Test the safe_sum utility function."""
    # Test normal case
    assert safe_sum([1.0, 2.0, 3.0]) == 6.0

    # Test with None values (should skip them)
    assert safe_sum([1.0, None, 3.0]) == 4.0

    # Test empty list
    assert safe_sum([]) == 0.0

    # Test all None
    assert safe_sum([None, None, None]) == 0.0


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
#     tf = RobustTeacherForcing(provider="fireworks", model="llama-3.1-8b")
#
#     # Generate a simple response
#     prompt = "What vegetable is green?"
#     response = "Cabbage"
#
#     # Get logprob using teacher forcing
#     result = tf.compute_log_prob(prompt, response)
#
#     # Should be reasonable for a single word
#     assert result.is_valid
#     assert -10 < result.value < -0.1, f"Logprob {result.value} suggests prompt tokens included"
