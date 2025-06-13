"""
Mock judge implementations for testing without external dependencies.

These mock implementations simulate the behavior of different judge types
without making actual API calls or loading models.
"""

import hashlib
import math
import random
from typing import List, Dict, Any, Optional, Union, Type
from dataclasses import dataclass

# Import the real Judge base class
try:
    from ...judge.judges import Judge as JudgeBase
except ImportError:
    # Fallback if import fails - create a mock Judge base class
    class JudgeBase:  # type: ignore[no-redef]
        def score(self, context: str, response: str) -> float:
            raise NotImplementedError

        def score_batch(
            self, samples: List[Dict[str, str]], disable_progress: bool = False
        ) -> List[float]:
            return [self.score(s["context"], s["response"]) for s in samples]


# Simple fallback classes for testing isolation
class _MockJudgeConfig:
    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        self.name = name or "mock"
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockAPIJudgeConfig(_MockJudgeConfig):
    pass


class _MockLocalJudgeConfig(_MockJudgeConfig):
    pass


@dataclass
class MockJudgeConfig:
    """Configuration for mock judge behavior."""

    # Scoring parameters
    base_score_range: tuple = (1.0, 10.0)  # Min/max possible scores
    score_variance: float = 1.0  # How much scores vary

    # Judge personality
    strictness: float = 0.5  # 0.0 = very lenient, 1.0 = very strict
    consistency: float = 0.8  # 0.0 = random scores, 1.0 = very consistent
    quality_sensitivity: float = 0.7  # How much judge cares about quality indicators

    # Biases (realistic judge biases)
    length_bias: float = 0.0  # Preference for longer/shorter responses (-1 to 1)
    formality_bias: float = 0.0  # Preference for formal/informal language
    creativity_bias: float = 0.0  # Preference for creative/conservative responses


class MockJudge(JudgeBase):
    """
    Base mock judge implementation.

    Provides deterministic but realistic scoring without external dependencies.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], _MockJudgeConfig],
        mock_config: Optional[MockJudgeConfig] = None,
    ):
        """Initialize mock judge."""
        if isinstance(config, dict):
            # Convert dict to config object
            # Ensure required fields are present
            if "name" not in config:
                config["name"] = "mock"
            if "template" not in config:
                config["template"] = "quick_judge"  # Use valid template

            judge_config = _MockJudgeConfig(**config)
            config = judge_config

        self.config = config
        self.mock_config = mock_config or MockJudgeConfig()

        # Setup judge personality based on config
        self._setup_judge_personality()

        # Cache for consistency
        self._score_cache: Dict[str, float] = {}

    def _setup_judge_personality(self) -> None:
        """Setup judge personality based on template and model."""
        template = getattr(self.config, "template", "quick_judge")
        model_name = getattr(self.config, "model_name", "default")

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
            elif "claude" in model_lower:
                self.mock_config.consistency = 0.85
                self.mock_config.creativity_bias = 0.2
            elif "gemini" in model_lower:
                self.mock_config.consistency = 0.8
                self.mock_config.quality_sensitivity = 0.7

    def _deterministic_random(self, context: str, response: str) -> random.Random:
        """Create deterministic random generator for consistent scoring."""
        combined = f"{context}||{response}||{getattr(self.config, 'template', '')}||{getattr(self.config, 'model_name', '')}"
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
        optimal_length = max(
            10, context_length * 2
        )  # Heuristic for good response length
        length_ratio = response_length / optimal_length
        length_score = 1.0 - abs(
            1.0 - min(length_ratio, 2.0)
        )  # Penalty for too short/long

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
            coherence_score = min(1.0, unique_ratio * 1.2)  # Slight bonus for variety
        else:
            coherence_score = 0.0

        # Formality (presence of formal indicators)
        formal_indicators = [
            "therefore",
            "furthermore",
            "however",
            "consequently",
            "moreover",
        ]
        informal_indicators = ["yeah", "ok", "cool", "awesome", "stuff"]

        formal_count = sum(1 for word in formal_indicators if word in response.lower())
        informal_count = sum(
            1 for word in informal_indicators if word in response.lower()
        )

        # Formality score from -1 (very informal) to 1 (very formal)
        formality_score = (formal_count - informal_count) / max(
            1, len(response.split()) / 10
        )
        formality_score = max(-1.0, min(1.0, formality_score))

        # Creativity (varied vocabulary and creative phrases)
        creative_indicators = [
            "imagine",
            "consider",
            "envision",
            "innovative",
            "unique",
            "creative",
        ]
        conservative_indicators = [
            "standard",
            "typical",
            "usual",
            "normal",
            "conventional",
        ]

        creative_count = sum(
            1 for word in creative_indicators if word in response.lower()
        )
        conservative_count = sum(
            1 for word in conservative_indicators if word in response.lower()
        )

        creativity_score = (creative_count - conservative_count) / max(
            1, len(response.split()) / 10
        )
        creativity_score = max(-1.0, min(1.0, creativity_score))

        return {
            "length_score": length_score,
            "relevance_score": relevance_score,
            "coherence_score": coherence_score,
            "formality_score": formality_score,
            "creativity_score": creativity_score,
            "response_length": response_length,
            "context_length": context_length,
        }

    def _calculate_base_score(self, context: str, response: str) -> float:
        """Calculate base score before biases and noise."""
        quality_metrics = self._analyze_response_quality(context, response)

        # Weighted combination of quality factors
        base_quality = (
            quality_metrics["relevance_score"] * 0.4  # Most important
            + quality_metrics["coherence_score"] * 0.3  # Second most important
            + quality_metrics["length_score"] * 0.2  # Length appropriateness
            + 0.5 * 0.1  # Base score for having a response
        )

        # Apply quality sensitivity
        adjusted_quality = (
            base_quality * self.mock_config.quality_sensitivity
            + (1 - self.mock_config.quality_sensitivity) * 0.5
        )

        # Apply strictness (strict judges give lower scores)
        strictness_adjustment = 1.0 - (self.mock_config.strictness * 0.3)
        adjusted_quality *= strictness_adjustment

        return adjusted_quality

    def _apply_biases(
        self, base_score: float, quality_metrics: Dict[str, float]
    ) -> float:
        """Apply judge-specific biases to the base score."""
        score = base_score

        # Length bias
        if self.mock_config.length_bias != 0:
            length_factor = (
                quality_metrics["response_length"] / 50.0
            )  # Normalize around 50 words
            length_adjustment = (
                self.mock_config.length_bias * (length_factor - 1.0) * 0.1
            )
            score += length_adjustment

        # Formality bias
        if self.mock_config.formality_bias != 0:
            formality_adjustment = (
                self.mock_config.formality_bias
                * quality_metrics["formality_score"]
                * 0.1
            )
            score += formality_adjustment

        # Creativity bias
        if self.mock_config.creativity_bias != 0:
            creativity_adjustment = (
                self.mock_config.creativity_bias
                * quality_metrics["creativity_score"]
                * 0.1
            )
            score += creativity_adjustment

        return score

    def _add_noise(self, score: float, rng: random.Random) -> float:
        """Add realistic noise to score based on consistency."""
        if self.mock_config.consistency >= 1.0:
            return score

        # Noise amount inversely related to consistency
        noise_amount = (
            1.0 - self.mock_config.consistency
        ) * self.mock_config.score_variance

        # Add Gaussian noise
        noise = rng.gauss(0, noise_amount * 0.1)  # Scale noise appropriately

        return score + noise

    def score(self, context: str, response: str) -> float:
        """Score a single context-response pair."""
        # Check cache for consistency
        cache_key = f"{hash(context)}:{hash(response)}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Calculate score
        rng = self._deterministic_random(context, response)

        # Calculate base score (0-1 range)
        base_score: float = self._calculate_base_score(context, response)

        # Analyze response for bias application
        quality_metrics = self._analyze_response_quality(context, response)

        # Apply biases
        biased_score: float = self._apply_biases(base_score, quality_metrics)

        # Add realistic noise
        final_score: float = self._add_noise(biased_score, rng)

        # Scale to target range and clamp
        min_score: float = float(self.mock_config.base_score_range[0])
        max_score: float = float(self.mock_config.base_score_range[1])
        scaled_score: float = min_score + (final_score * (max_score - min_score))
        clamped_score: float = max(min_score, min(max_score, scaled_score))

        # Cache result
        self._score_cache[cache_key] = clamped_score

        return clamped_score

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[float]:
        """Score a batch of context-response pairs."""
        return [self.score(sample["context"], sample["response"]) for sample in samples]


class MockAPIJudge(MockJudge):
    """
    Mock API-based judge implementation.

    Simulates API judge behavior with provider-specific characteristics.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], _MockAPIJudgeConfig],
        mock_config: Optional[MockJudgeConfig] = None,
    ):
        """Initialize mock API judge."""
        if isinstance(config, dict):
            api_config = _MockAPIJudgeConfig()
            for key, value in config.items():
                setattr(api_config, key, value)
            config = api_config

        super().__init__(config, mock_config)
        self._setup_api_personality()

    def _setup_api_personality(self) -> None:
        """Setup personality based on API provider."""
        provider = getattr(self.config, "provider", "openai")

        if provider == "openai":
            # OpenAI models tend to be consistent and quality-focused
            self.mock_config.consistency = 0.85
            self.mock_config.quality_sensitivity = 0.8
            self.mock_config.strictness = 0.6
        elif provider == "anthropic":
            # Claude tends to be creative and thoughtful
            self.mock_config.consistency = 0.8
            self.mock_config.creativity_bias = 0.2
            self.mock_config.formality_bias = 0.1
        elif provider == "google":
            # Google models tend to be factual and consistent
            self.mock_config.consistency = 0.82
            self.mock_config.quality_sensitivity = 0.75
            self.mock_config.strictness = 0.55

    def score(self, context: str, response: str) -> float:
        """Score with simulated API latency."""
        # Simulate very small API delay for realism
        import time

        time.sleep(0.001)  # 1ms delay

        return super().score(context, response)

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[float]:
        """Score batch with simulated API processing."""
        # Simulate realistic API batch processing time
        import time

        if len(samples) > 5:
            time.sleep(0.01)  # 10ms for larger batches

        return super().score_batch(samples, disable_progress)


class MockLocalJudge(MockJudge):
    """
    Mock local model-based judge implementation.

    Simulates local model behavior without loading actual models.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], _MockLocalJudgeConfig],
        mock_config: Optional[MockJudgeConfig] = None,
    ):
        """Initialize mock local judge."""
        if isinstance(config, dict):
            local_config = _MockLocalJudgeConfig()
            for key, value in config.items():
                setattr(local_config, key, value)
            config = local_config

        super().__init__(config, mock_config)
        self._setup_local_personality()

    def _setup_local_personality(self) -> None:
        """Setup personality based on local model characteristics."""
        model_name = getattr(self.config, "model_name", "")

        if isinstance(model_name, str):
            model_lower = model_name.lower()

            # Smaller models tend to be less consistent
            if "small" in model_lower or "tiny" in model_lower or "7b" in model_lower:
                self.mock_config.consistency = 0.6
                self.mock_config.quality_sensitivity = 0.5
                self.mock_config.score_variance = 1.5
            elif "13b" in model_lower or "medium" in model_lower:
                self.mock_config.consistency = 0.7
                self.mock_config.quality_sensitivity = 0.6
            elif "70b" in model_lower or "large" in model_lower:
                self.mock_config.consistency = 0.8
                self.mock_config.quality_sensitivity = 0.75

            # Model family characteristics
            if "llama" in model_lower:
                self.mock_config.creativity_bias = 0.1
                self.mock_config.length_bias = 0.2  # Llama tends to be verbose
            elif "mistral" in model_lower:
                self.mock_config.formality_bias = 0.1
                self.mock_config.strictness = 0.7
            elif "alpaca" in model_lower:
                self.mock_config.creativity_bias = -0.1  # More conservative
                self.mock_config.formality_bias = 0.2

    def score(self, context: str, response: str) -> float:
        """Score with simulated local model processing."""
        # Simulate small processing delay for local inference
        import time

        time.sleep(0.002)  # 2ms delay

        return super().score(context, response)


# Factory function for easy judge creation
def create_mock_judge(
    judge_type: str,
    config: Optional[Dict[str, Any]] = None,
    mock_config: Optional[MockJudgeConfig] = None,
) -> MockJudge:
    """
    Factory function to create appropriate mock judge.

    Args:
        judge_type: Type of judge ('openai', 'anthropic', 'google', 'local', 'prometheus', etc.)
        config: Judge configuration
        mock_config: Mock-specific configuration

    Returns:
        Appropriate mock judge instance
    """
    config = config or {}

    # Ensure name is set in config
    if "name" not in config:
        config["name"] = judge_type

    # Set defaults based on judge type
    if judge_type in ["openai", "anthropic", "google"]:
        config.setdefault("provider", judge_type)
        config.setdefault("model_name", f"{judge_type}-default")

        # Use mock config classes for consistency
        api_config = _MockAPIJudgeConfig(**config)
        return MockAPIJudge(api_config, mock_config)

    elif judge_type in ["local", "huggingface", "prometheus"]:
        config.setdefault("model_name", "mock-local-judge")
        config.setdefault("device", "cpu")

        # Use mock config classes for consistency
        local_config = _MockLocalJudgeConfig(**config)
        return MockLocalJudge(local_config, mock_config)

    else:
        # Default to base mock judge
        base_config = _MockJudgeConfig(**config)
        return MockJudge(base_config, mock_config)
