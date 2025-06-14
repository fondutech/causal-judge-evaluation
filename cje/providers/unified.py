"""
Unified LLM provider interface - simpler and more maintainable.
"""

from typing import Dict, List, Tuple, Optional, Any, Protocol, Union
import os
from abc import ABC, abstractmethod
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMProvider(Protocol):
    """Simple protocol for LLM providers."""

    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Tuple[str, Optional[float]]:
        """Return (response, log_prob) tuple."""
        ...

    def score(self, prompt: str, response: str, **kwargs: Any) -> float:
        """Return log probability of response given prompt."""
        ...


class UnifiedProvider:
    """Single entry point for all LLM providers."""

    def __init__(self, provider: str, model: str, **kwargs: Any):
        self.provider = provider.lower()
        self.model = model
        self.kwargs = kwargs
        self.client: Union[openai.OpenAI, anthropic.Anthropic]

        # Initialize appropriate client
        if self.provider == "openai":
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), **kwargs.get("client_kwargs", {})
            )
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                **kwargs.get("client_kwargs", {}),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        return_logprobs: bool = False,
        **kwargs: Any,
    ) -> Tuple[str, Optional[float]]:
        """Unified completion interface."""

        if self.provider == "openai":
            assert isinstance(self.client, openai.OpenAI)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=return_logprobs,
                **kwargs,
            )

            text = response.choices[0].message.content or ""
            logprob = None

            if return_logprobs and response.choices[0].logprobs:
                # Sum token log probs
                logprob = sum(
                    t.logprob
                    for t in response.choices[0].logprobs.content
                    if t.logprob is not None
                )

            return text, logprob

        elif self.provider == "anthropic":
            assert isinstance(self.client, anthropic.Anthropic)
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            text = response.content[0].text if response.content else ""
            # Anthropic doesn't provide log probs in standard API
            return text, None

        else:
            raise ValueError(f"Provider {self.provider} not implemented")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def score(self, prompt: str, response: str, **kwargs: Any) -> float:
        """Score a response - return log probability."""

        if self.provider == "openai":
            # Use logprobs parameter to score
            full_prompt = f"{prompt}\n\nAssistant: {response}"
            assert isinstance(self.client, openai.OpenAI)

            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                temperature=0,
                max_tokens=1,  # Just need scoring
                logprobs=True,
                **kwargs,
            )

            # Extract log probability
            if result.choices[0].logprobs and result.choices[0].logprobs.content:
                total: float = sum(
                    t.logprob
                    for t in result.choices[0].logprobs.content
                    if t.logprob is not None
                )
                return total
            else:
                return -100.0  # Fallback for errors

        else:
            # For providers without native scoring, use a heuristic
            # This is a limitation we should document
            return -100.0

    def judge(
        self, context: str, response: str, template: str = "default", **kwargs: Any
    ) -> float:
        """Judge a response - return normalized score."""

        # Simple default template
        if template == "default":
            judge_prompt = f"""Rate the quality of this response on a scale of 0-10.

Context: {context}

Response: {response}

Provide only a number between 0 and 10."""
        else:
            # Could load from template library
            judge_prompt = template.format(context=context, response=response)

        score_text, _ = self.complete(judge_prompt, temperature=0, max_tokens=10)

        # Extract numeric score
        try:
            score = float(score_text.strip().split()[0])
            return max(0, min(10, score)) / 10.0  # Normalize to [0, 1]
        except:
            return 0.5  # Default middle score


def create_provider(config: Dict[str, Any]) -> UnifiedProvider:
    """Factory function to create providers from config."""
    return UnifiedProvider(
        provider=config["provider"],
        model=config["model_name"],
        **config.get("kwargs", {}),
    )
