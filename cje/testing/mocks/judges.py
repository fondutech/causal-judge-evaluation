"""
Unified mock judge implementations that return JudgeScore with uncertainty.

These mock implementations simulate the behavior of different judge types
without making actual API calls or loading models.
"""

import hashlib
import math
import random
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Import unified judge base and schemas
from ...judge.judges import Judge, DeterministicJudge
from ...judge.schemas import JudgeScore


@dataclass
class MockJudgeConfig:
    """Configuration for mock judge behavior."""

    # Scoring parameters
    base_score_range: tuple = (0.0, 1.0)  # Now 0-1 range for unified system
    score_variance: float = 0.1  # Base variance for scores

    # Uncertainty parameters
    uncertainty_method: str = "structured"  # deterministic, structured, monte_carlo
    base_uncertainty: float = 0.02  # Base uncertainty level
    uncertainty_scaling: float = 1.0  # How much uncertainty varies

    # Judge personality
    strictness: float = 0.5  # 0.0 = very lenient, 1.0 = very strict
    consistency: float = 0.8  # 0.0 = random scores, 1.0 = very consistent
    quality_sensitivity: float = 0.7  # How much judge cares about quality indicators

    # Biases (realistic judge biases)
    length_bias: float = 0.0  # Preference for longer/shorter responses (-1 to 1)
    formality_bias: float = 0.0  # Preference for formal/informal language
    creativity_bias: float = 0.0  # Preference for creative/conservative responses


class MockJudge(Judge):
    """
    Unified mock judge that returns JudgeScore with uncertainty.

    Provides deterministic but realistic scoring without external dependencies.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        mock_config: Optional[MockJudgeConfig] = None,
    ):
        """Initialize mock judge."""
        self.config = config
        self.mock_config = mock_config or MockJudgeConfig()

        # Setup judge personality based on config
        self._setup_judge_personality()

        # Cache for consistency
        self._score_cache: Dict[str, JudgeScore] = {}

    def _setup_judge_personality(self) -> None:
        """Setup judge personality based on template and model."""
        template = self.config.get("template", "quick_judge")
        model_name = self.config.get("model_name", "default")

        # Adjust personality based on template
        if "strict" in template.lower():
            self.mock_config.strictness = 0.8
            self.mock_config.quality_sensitivity = 0.9
        elif "lenient" in template.lower():
            self.mock_config.strictness = 0.3
            self.mock_config.quality_sensitivity = 0.5
        elif "creative" in template.lower():
            self.mock_config.creativity_bias = 0.5
            self.mock_config.formality_bias = -0.3
        elif "formal" in template.lower():
            self.mock_config.formality_bias = 0.5
            self.mock_config.creativity_bias = -0.2

        # Adjust based on model (if specified)
        if model_name and isinstance(model_name, str):
            model_lower = model_name.lower()
            if "gpt-4" in model_lower:
                self.mock_config.consistency = 0.9
                self.mock_config.quality_sensitivity = 0.8
                self.mock_config.base_uncertainty = 0.01  # GPT-4 is confident
            elif "claude" in model_lower:
                self.mock_config.consistency = 0.85
                self.mock_config.creativity_bias = 0.2
                self.mock_config.base_uncertainty = 0.015
            elif "gemini" in model_lower:
                self.mock_config.consistency = 0.8
                self.mock_config.quality_sensitivity = 0.7
                self.mock_config.base_uncertainty = 0.02

    def _deterministic_random(self, context: str, response: str) -> random.Random:
        """Create deterministic random generator for consistent scoring."""
        combined = f"{context}||{response}||{self.config.get('template', '')}||{self.config.get('model_name', '')}"
        seed = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        return random.Random(seed)

    def _analyze_response_quality(
        self, context: str, response: str
    ) -> Dict[str, float]:
        """Analyze response for various quality indicators."""
        # Basic metrics
        response_length = len(response.split())
        context_length = len(context.split())

        # Length appropriateness (0-1 score)
        optimal_length = max(10, context_length * 2)
        length_ratio = response_length / optimal_length
        length_score = 1.0 - abs(1.0 - min(length_ratio, 2.0))

        # Relevance (basic word overlap)
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        if len(context_words) > 0:
            relevance_score = len(context_words & response_words) / len(context_words)
        else:
            relevance_score = 0.5

        # Coherence (anti-repetition)
        response_words_list = response.lower().split()
        if len(response_words_list) > 0:
            unique_ratio = len(set(response_words_list)) / len(response_words_list)
            coherence_score = min(1.0, unique_ratio * 1.2)
        else:
            coherence_score = 0.0

        # Overall quality score for uncertainty estimation
        overall_quality = (
            relevance_score * 0.4 + coherence_score * 0.3 + length_score * 0.3
        )

        return {
            "length_score": length_score,
            "relevance_score": relevance_score,
            "coherence_score": coherence_score,
            "overall_quality": overall_quality,
            "response_length": response_length,
            "context_length": context_length,
        }

    def _calculate_mean_score(self, context: str, response: str) -> float:
        """Calculate mean score."""
        quality_metrics = self._analyze_response_quality(context, response)

        # Base quality
        base_quality = quality_metrics["overall_quality"]

        # Apply quality sensitivity
        adjusted_quality = (
            base_quality * self.mock_config.quality_sensitivity
            + (1 - self.mock_config.quality_sensitivity) * 0.5
        )

        # Apply strictness
        strictness_adjustment = 1.0 - (self.mock_config.strictness * 0.3)
        adjusted_quality *= strictness_adjustment

        # Add biases
        rng = self._deterministic_random(context, response)
        if self.mock_config.consistency < 1.0:
            noise = rng.gauss(0, (1 - self.mock_config.consistency) * 0.1)
            adjusted_quality += noise

        # Clamp to [0, 1]
        return max(0.0, min(1.0, adjusted_quality))

    def _calculate_variance(self, mean: float, context: str, response: str) -> float:
        """Calculate variance based on uncertainty method."""
        if self.mock_config.uncertainty_method == "deterministic":
            return 0.0

        quality_metrics = self._analyze_response_quality(context, response)

        # Base uncertainty
        base_var = self.mock_config.base_uncertainty

        # Uncertainty increases for:
        # 1. Scores near 0.5 (maximum uncertainty)
        distance_from_middle = abs(mean - 0.5)
        uncertainty_from_score = 1.0 - 2 * distance_from_middle

        # 2. Low quality responses
        quality_factor = 1.0 - quality_metrics["overall_quality"]

        # 3. Short responses (less information)
        length_factor = 1.0 / (1.0 + quality_metrics["response_length"] / 20)

        # Combine factors
        total_uncertainty = (
            base_var
            * (
                1.0
                + uncertainty_from_score * 0.5
                + quality_factor * 0.3
                + length_factor * 0.2
            )
            * self.mock_config.uncertainty_scaling
        )

        # Ensure variance doesn't exceed theoretical maximum
        max_variance = mean * (1 - mean)  # Variance of Bernoulli
        return min(total_uncertainty, max_variance, 0.25)

    def score(self, context: str, response: str) -> JudgeScore:
        """Score a single context-response pair with uncertainty."""
        # Check cache for consistency
        cache_key = f"{hash(context)}:{hash(response)}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Calculate mean score
        mean = self._calculate_mean_score(context, response)

        # Calculate variance
        variance = self._calculate_variance(mean, context, response)

        # Create JudgeScore
        score = JudgeScore(mean=mean, variance=variance)

        # Cache result
        self._score_cache[cache_key] = score

        return score

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Score a batch of context-response pairs."""
        return [self.score(sample["context"], sample["response"]) for sample in samples]


class DeterministicMockJudge(MockJudge):
    """Mock judge that always returns zero variance."""

    def __init__(
        self, config: Dict[str, Any], mock_config: Optional[MockJudgeConfig] = None
    ):
        if mock_config is None:
            mock_config = MockJudgeConfig()
        mock_config.uncertainty_method = "deterministic"
        super().__init__(config, mock_config)


class MCMockJudge(MockJudge):
    """Mock judge that simulates Monte Carlo uncertainty estimation."""

    def __init__(
        self,
        config: Dict[str, Any],
        mock_config: Optional[MockJudgeConfig] = None,
        n_samples: int = 5,
    ):
        super().__init__(config, mock_config)
        self.n_samples = n_samples
        self.mock_config.uncertainty_method = "monte_carlo"

    def score(self, context: str, response: str) -> JudgeScore:
        """Score using simulated Monte Carlo sampling."""
        # Get base score
        base_score = super().score(context, response)

        # Simulate MC sampling by adding controlled noise
        rng = self._deterministic_random(context, response)
        samples = []

        for i in range(self.n_samples):
            # Add temperature-based noise
            noise = rng.gauss(0, 0.05)  # Simulate temperature effect
            sample = max(0, min(1, base_score.mean + noise))
            samples.append(sample)

        # Calculate empirical mean and variance
        import numpy as np

        mean = np.mean(samples)
        variance = np.var(samples)

        return JudgeScore(mean=float(mean), variance=float(variance))


# Convenience classes with preset personalities
class LenientJudge(MockJudge):
    """Lenient mock judge - gives higher scores with low variance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        mock_config = MockJudgeConfig(
            strictness=0.2,
            quality_sensitivity=0.4,
            base_uncertainty=0.01,
            consistency=0.9,
        )
        super().__init__(config, mock_config)


class HarshJudge(MockJudge):
    """Harsh mock judge - gives lower scores with higher variance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        mock_config = MockJudgeConfig(
            strictness=0.9,
            quality_sensitivity=0.95,
            base_uncertainty=0.03,
            consistency=0.7,
        )
        super().__init__(config, mock_config)


class NoisyJudge(MockJudge):
    """Noisy mock judge - inconsistent scores with high variance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        mock_config = MockJudgeConfig(
            consistency=0.3,
            base_uncertainty=0.05,
            uncertainty_scaling=2.0,
            score_variance=0.2,
        )
        super().__init__(config, mock_config)


class RandomJudge(MockJudge):
    """Random mock judge - completely random scores."""

    def score(self, context: str, response: str) -> JudgeScore:
        """Return random score with high uncertainty."""
        rng = self._deterministic_random(context, response)
        mean = rng.random()
        # High variance for random scores
        variance = 0.08 + rng.random() * 0.08
        return JudgeScore(mean=mean, variance=variance)


class ConstantJudge(DeterministicJudge):
    """Constant mock judge - always returns the same score."""

    def __init__(self, constant_score: float = 0.7):
        self.constant_score = constant_score

    def score_value(self, context: str, response: str) -> float:
        """Always return constant score."""
        return self.constant_score


# Factory function for easy judge creation
def create_mock_judge(
    judge_type: str,
    config: Optional[Dict[str, Any]] = None,
    mock_config: Optional[MockJudgeConfig] = None,
    uncertainty_method: str = "structured",
) -> Judge:
    """
    Factory function to create appropriate mock judge.

    Args:
        judge_type: Type of judge ('lenient', 'harsh', 'noisy', 'random', 'constant', etc.)
        config: Judge configuration
        mock_config: Mock-specific configuration
        uncertainty_method: How to estimate uncertainty

    Returns:
        Appropriate mock judge instance
    """
    config = config or {}

    # Set defaults
    config.setdefault("name", f"mock-{judge_type}")
    config.setdefault("template", "quick_judge")

    # Create based on type
    if judge_type == "lenient":
        return LenientJudge(config)
    elif judge_type == "harsh":
        return HarshJudge(config)
    elif judge_type == "noisy":
        return NoisyJudge(config)
    elif judge_type == "random":
        return RandomJudge(config)
    elif judge_type == "constant":
        score = config.get("constant_score", 0.7)
        return ConstantJudge(score)
    elif judge_type == "deterministic":
        return DeterministicMockJudge(config, mock_config)
    elif judge_type == "monte_carlo":
        n_samples = config.get("mc_samples", 5)
        return MCMockJudge(config, mock_config, n_samples)
    else:
        # Default mock judge
        if mock_config is None:
            mock_config = MockJudgeConfig(uncertainty_method=uncertainty_method)
        return MockJudge(config, mock_config)
