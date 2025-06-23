"""Confidence Interval Judge - Explicit uncertainty through confidence intervals."""

from typing import Optional
from .api_judge import APIJudge, APIJudgeConfig
from .schemas import JudgeScore
import logging

logger = logging.getLogger(__name__)


class ConfidenceIntervalJudge(APIJudge):
    """Judge that explicitly asks for confidence intervals to quantify uncertainty.

    This judge:
    1. Uses a template that asks for score + 95% CI
    2. Parses the CI from the response
    3. Calculates variance from the CI width

    The variance calculation assumes a normal distribution where:
    - 95% CI = mean ± 1.96*σ
    - Therefore: σ = CI_width / 3.92
    - Variance = σ² (scaled to 0-1 range)
    """

    def __init__(self, config: APIJudgeConfig):
        # Force the confidence_interval template
        if config.template not in [
            "confidence_interval",
            "ci_judge",
            "uncertainty_judge",
        ]:
            logger.warning(
                f"CI Judge should use CI template, not '{config.template}'. "
                f"Switching to 'confidence_interval'"
            )
            config.template = "confidence_interval"

        # Use temperature=0 for consistency
        if config.temperature > 0:
            logger.warning(
                f"CI Judge works best with temperature=0, not {config.temperature}"
            )
            config.temperature = 0.0

        # Use the JudgeScoreWithCI schema for structured output
        config.structured_output_schema = "JudgeScoreWithCI"
        config.use_structured_output = True

        super().__init__(config)

    # No need to override _score_async - the structured output handles everything!
