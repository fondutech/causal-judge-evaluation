"""
Mock policy runners for testing without external dependencies.

These mock implementations simulate the behavior of PolicyRunner and APIPolicyRunner
without loading actual models or making API calls.
"""

import hashlib
import random
import math
from typing import List, Tuple, Optional, Any, Union, Protocol
from dataclasses import dataclass


@dataclass
class MockModelConfig:
    """Configuration for mock model behavior."""

    # Response generation parameters
    avg_response_length: int = 50
    response_length_variance: int = 20
    vocab_size: int = 50000

    # Log probability simulation
    base_logp_per_token: float = -3.0
    logp_variance: float = 1.0

    # Model personality parameters (affects response style)
    creativity: float = 0.5  # 0.0 = very conservative, 1.0 = very creative
    verbosity: float = 0.5  # 0.0 = terse, 1.0 = verbose
    quality: float = 0.8  # 0.0 = poor responses, 1.0 = excellent responses

    # Temperature effects
    temperature_sensitivity: float = 1.0  # How much temperature affects randomness


class PolicyRunnerProtocol(Protocol):
    """Protocol defining the interface for policy runners."""

    def generate_with_logp(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Tuple[str, float, Any]]:
        """Generate responses with log probabilities."""
        ...

    def log_prob(self, context: str, response: str, **kwargs: Any) -> float:
        """Calculate log probability of response given context."""
        ...


class MockPolicyRunner(PolicyRunnerProtocol):
    """
    Mock implementation of PolicyRunner for testing.

    Simulates transformer model behavior without loading actual models.
    Provides deterministic but realistic-looking responses and log probabilities.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
        text_format: str = "standard",
        config: Optional[MockModelConfig] = None,
    ):
        """Initialize mock policy runner."""
        self.model_name = model_name
        self.device = device or "cpu"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.user_message_template = user_message_template
        self.text_format = text_format

        # Mock model configuration
        self.config = config or MockModelConfig()
        self._setup_model_personality()

        # Deterministic random seed based on model name for consistency
        self._seed = int(hashlib.md5(model_name.encode()).hexdigest()[:8], 16)

    def _setup_model_personality(self) -> None:
        """Setup model personality based on model name."""
        model_lower = self.model_name.lower()

        # Adjust personality based on common model characteristics
        if "gpt-4" in model_lower:
            self.config.quality = 0.9
            self.config.verbosity = 0.7
            self.config.creativity = 0.6
        elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            self.config.quality = 0.8
            self.config.verbosity = 0.6
            self.config.creativity = 0.5
        elif "claude" in model_lower:
            self.config.quality = 0.85
            self.config.verbosity = 0.8
            self.config.creativity = 0.7
        elif "gemini" in model_lower:
            self.config.quality = 0.8
            self.config.verbosity = 0.6
            self.config.creativity = 0.6
        elif "llama" in model_lower:
            self.config.quality = 0.75
            self.config.verbosity = 0.5
            self.config.creativity = 0.6
        elif "tiny" in model_lower or "small" in model_lower:
            # Small/test models
            self.config.quality = 0.4
            self.config.verbosity = 0.3
            self.config.creativity = 0.3
            self.config.avg_response_length = 20

    def _deterministic_random(
        self, context: str, extra_seed: str = ""
    ) -> random.Random:
        """Create deterministic random generator for consistent testing."""
        combined_seed = f"{self.model_name}:{context}:{extra_seed}"
        seed = int(hashlib.md5(combined_seed.encode()).hexdigest()[:8], 16)
        return random.Random(seed)

    def _generate_response(self, context: str, rng: random.Random) -> str:
        """Generate a mock response based on context and model personality."""
        # Determine response length based on context and model config
        base_length = self.config.avg_response_length

        # Adjust length based on context length and content
        context_length = len(context.split())
        length_factor = min(
            2.0, context_length / 20.0
        )  # Longer contexts get longer responses

        # Apply verbosity and temperature effects
        verbosity_factor = 0.5 + self.config.verbosity
        temp_factor = 1.0 + (
            self.temperature * self.config.temperature_sensitivity * 0.5
        )

        target_length = int(
            base_length * length_factor * verbosity_factor * temp_factor
        )
        target_length = max(5, min(target_length, self.max_new_tokens))

        # Add some randomness to length
        variance = int(self.config.response_length_variance * temp_factor)
        actual_length = max(
            5, rng.randint(target_length - variance, target_length + variance)
        )

        # Generate response based on context type and content
        response_parts = []

        # Analyze context for response type
        context_lower = context.lower()

        if any(
            word in context_lower
            for word in ["what", "how", "why", "explain", "describe"]
        ):
            # Explanatory response
            response_parts.append(
                self._generate_explanation(context, actual_length, rng)
            )
        elif any(
            word in context_lower for word in ["write", "create", "generate", "compose"]
        ):
            # Creative response
            response_parts.append(
                self._generate_creative_content(context, actual_length, rng)
            )
        elif "?" in context:
            # Question answering
            response_parts.append(self._generate_answer(context, actual_length, rng))
        else:
            # General response
            response_parts.append(
                self._generate_general_response(context, actual_length, rng)
            )

        response = " ".join(response_parts)

        # Ensure minimum quality
        if self.config.quality < 0.5 and rng.random() < 0.3:
            # Sometimes generate lower quality responses for poor models
            response = self._degrade_response_quality(response, rng)

        return response[: self.max_new_tokens * 4]  # Rough character limit

    def _generate_explanation(
        self, context: str, target_length: int, rng: random.Random
    ) -> str:
        """Generate explanatory response."""
        explanations = [
            "This is a complex topic that involves several key concepts.",
            "To understand this, we need to consider multiple perspectives.",
            "The main idea centers around the fundamental principle that",
            "There are several important factors to consider here.",
            "This concept can be broken down into several components.",
            "The underlying mechanism works through a series of steps.",
            "Research has shown that this phenomenon occurs when",
            "The key insight is that this process involves",
        ]

        starters = [
            "First, it's important to understand that",
            "Additionally, we should note that",
            "Furthermore, the evidence suggests that",
            "In practical terms, this means that",
            "From a technical perspective,",
            "It's worth mentioning that",
            "Another crucial aspect is that",
            "The implications of this are that",
        ]

        response_parts = [rng.choice(explanations)]
        current_length = len(response_parts[0].split())

        while current_length < target_length * 0.8:
            starter = rng.choice(starters)
            continuation = "this demonstrates the importance of careful analysis and consideration."
            if rng.random() < self.config.creativity:
                continuation = "the interconnected nature of these systems creates emergent properties."

            new_part = f"{starter} {continuation}"
            response_parts.append(new_part)
            current_length += len(new_part.split())

        return " ".join(response_parts)

    def _generate_creative_content(
        self, context: str, target_length: int, rng: random.Random
    ) -> str:
        """Generate creative content."""
        if self.config.creativity > 0.7:
            openers = [
                "Imagine a world where",
                "Picture this scenario:",
                "Consider the possibility that",
                "Let me paint you a picture of",
                "Envision a future where",
                "What if we could",
            ]
        else:
            openers = [
                "Here is a response about",
                "This topic involves",
                "The main points are",
                "To address this request,",
                "The following information is relevant:",
            ]

        opener = rng.choice(openers)

        # Generate content based on creativity level
        if self.config.creativity > 0.6:
            content = "innovative ideas merge with practical applications to create unprecedented opportunities for growth and discovery."
        else:
            content = "standard approaches and established methodologies provide reliable results for most use cases."

        return f"{opener} {content} This approach ensures both effectiveness and reliability in the implementation process."

    def _generate_answer(
        self, context: str, target_length: int, rng: random.Random
    ) -> str:
        """Generate answer to a question."""
        direct_answers = [
            "The answer depends on several factors.",
            "Based on available information,",
            "From what we know,",
            "The evidence suggests that",
            "Generally speaking,",
            "In most cases,",
            "According to current understanding,",
        ]

        answer_start = rng.choice(direct_answers)

        explanations = [
            "this involves a careful balance of competing priorities and considerations.",
            "the underlying principles guide us toward an optimal solution.",
            "multiple variables interact to produce the observed outcomes.",
            "established best practices recommend a systematic approach.",
            "the key factors include timing, resources, and strategic alignment.",
        ]

        explanation = rng.choice(explanations)

        return f"{answer_start} {explanation}"

    def _generate_general_response(
        self, context: str, target_length: int, rng: random.Random
    ) -> str:
        """Generate general response."""
        responses = [
            "This is an interesting topic that deserves careful consideration.",
            "There are multiple ways to approach this subject matter.",
            "The context provides valuable insights into the underlying issues.",
            "This situation requires a thoughtful and balanced response.",
            "Understanding the nuances here is crucial for effective analysis.",
        ]

        return (
            rng.choice(responses)
            + " The key is to maintain focus on the core objectives while remaining flexible in implementation."
        )

    def _degrade_response_quality(self, response: str, rng: random.Random) -> str:
        """Degrade response quality for low-quality models."""
        # Add some repetition or unclear phrasing
        degradations = [
            lambda r: r + " " + r.split(".")[0] + ".",  # Repeat first sentence
            lambda r: (
                r.replace("the", "the the") if rng.random() < 0.3 else r
            ),  # Add redundancy
            lambda r: (
                r.replace(".", ". Um,") if rng.random() < 0.2 else r
            ),  # Add hesitation
        ]

        for degradation in degradations:
            if rng.random() < 0.4:  # 40% chance of each degradation
                response = degradation(response)

        return response

    def _calculate_logp(self, context: str, response: str) -> float:
        """Calculate mock log probability for a response given context."""
        # Create deterministic calculation based on content
        combined = f"{context}||{response}"

        # Calculate base log probability
        response_tokens = len(response.split())
        if response_tokens == 0:
            return -100.0

        # Base calculation using model config
        base_logp = self.config.base_logp_per_token * response_tokens

        # Adjust based on response quality indicators
        quality_adjustments = 0.0

        # Length appropriateness (not too short, not too long)
        optimal_length = len(context.split()) * 2  # Rough heuristic
        length_ratio = response_tokens / max(1, optimal_length)
        if 0.5 <= length_ratio <= 2.0:
            quality_adjustments += 0.5  # Appropriate length bonus
        else:
            quality_adjustments -= abs(length_ratio - 1.0)  # Penalty for poor length

        # Repetition penalty
        words = response.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            repetition_ratio = len(unique_words) / len(words)
            quality_adjustments += (repetition_ratio - 0.5) * 2.0

        # Context relevance (very basic heuristic)
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        overlap = len(context_words & response_words)
        relevance_score = overlap / max(1, len(context_words))
        quality_adjustments += relevance_score * 1.0

        # Temperature effects (higher temperature = higher variance)
        if self.temperature > 0:
            # Use deterministic "randomness" based on content hash
            content_hash = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
            normalized_hash = (content_hash % 10000) / 10000.0  # 0-1 range

            # Apply temperature-based variance
            temp_variance = self.config.logp_variance * self.temperature
            temp_adjustment = (normalized_hash - 0.5) * temp_variance * 2
            quality_adjustments += temp_adjustment

        # Model quality effects
        quality_adjustment = (self.config.quality - 0.5) * 2.0

        final_logp = base_logp + quality_adjustments + quality_adjustment

        # Ensure reasonable bounds
        return max(-100.0, min(-0.1, final_logp))

    def generate_with_logp(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Tuple[str, float, Any]]:
        """Generate responses with log probabilities."""
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        return_token_logprobs = kwargs.get("return_token_logprobs", False)

        # Temporarily update temperature for generation
        original_temp = self.temperature
        self.temperature = temperature if temperature is not None else self.temperature

        results: List[Tuple[str, float, Any]] = []
        for prompt in prompts:
            rng = self._deterministic_random(prompt, f"generate_{temperature}_{top_p}")

            # Generate response
            response = self._generate_response(prompt, rng)

            # Calculate log probability
            logp = self._calculate_logp(prompt, response)

            if return_token_logprobs:
                # Generate mock token-level log probabilities
                tokens = response.split()
                token_logps: List[float] = []
                for i, token in enumerate(tokens):
                    token_context = f"{prompt} {' '.join(tokens[:i])}"
                    token_logp = self._calculate_logp(token_context, token) / max(
                        1, len(token)
                    )
                    token_logps.append(float(token_logp))
                results.append((response, float(logp), token_logps))
            else:
                results.append((response, float(logp), None))

        # Restore original temperature
        self.temperature = original_temp

        return results

    def log_prob(self, context: str, response: str, **kwargs: Any) -> float:
        """Calculate log probability of response given context."""
        # Use provided parameters or fall back to instance defaults
        temperature = kwargs.get("temperature", self.temperature)

        # Temporarily update temperature for calculation
        original_temp = self.temperature
        self.temperature = temperature if temperature is not None else self.temperature

        logp = self._calculate_logp(context, response)

        debug = kwargs.get("debug", False)
        if debug:
            print(
                f"MockPolicyRunner.log_prob: context='{context[:50]}...', "
                f"response='{response[:50]}...', logp={logp}"
            )

        # Restore original temperature
        self.temperature = original_temp

        return logp


class MockAPIPolicyRunner(PolicyRunnerProtocol):
    """
    Mock implementation of APIPolicyRunner for testing.

    Simulates API-based model behavior without making actual API calls.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
        config: Optional[MockModelConfig] = None,
    ):
        """Initialize mock API policy runner."""
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or "mock_api_key"
        self.base_url = base_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.user_message_template = user_message_template

        # Use underlying mock policy runner for actual generation
        self._mock_runner = MockPolicyRunner(
            model_name=f"{provider}:{model_name}",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            user_message_template=user_message_template,
            config=config,
        )

        # Set up provider-specific behavior
        self._setup_provider_personality()

    def _setup_provider_personality(self) -> None:
        """Setup provider-specific model behavior."""
        if self.provider == "openai":
            if "gpt-4" in self.model_name:
                self._mock_runner.config.quality = 0.9
                self._mock_runner.config.verbosity = 0.7
            elif "gpt-3.5" in self.model_name:
                self._mock_runner.config.quality = 0.8
                self._mock_runner.config.verbosity = 0.6
        elif self.provider == "anthropic":
            self._mock_runner.config.quality = 0.85
            self._mock_runner.config.verbosity = 0.8
            self._mock_runner.config.creativity = 0.7
        elif self.provider == "google":
            self._mock_runner.config.quality = 0.8
            self._mock_runner.config.verbosity = 0.6

    def generate_with_logp(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Tuple[str, float, Any]]:
        """Generate responses with log probabilities via mock API."""
        # Simulate API latency with deterministic delay simulation
        import time

        if len(prompts) > 10:  # Only for large batches
            time.sleep(0.01)  # Very small delay to simulate processing

        return self._mock_runner.generate_with_logp(prompts=prompts, **kwargs)

    def log_prob(self, context: str, response: str, **kwargs: Any) -> float:
        """Calculate log probability via mock API."""
        return self._mock_runner.log_prob(context=context, response=response, **kwargs)
