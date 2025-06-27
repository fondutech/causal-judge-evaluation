"""
Local/HF policy wrapper that inherits from BasePolicy.

Wraps PolicyRunner to provide BasePolicy interface.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

from .base_policy import BasePolicy
from .policy import PolicyRunner
from ..types import LogProbResult

logger = logging.getLogger(__name__)


class LocalPolicy(BasePolicy):
    """
    Wrapper for PolicyRunner that provides BasePolicy interface.

    This allows local HF models to be used in the same way as API models.
    """

    def __init__(
        self,
        model_name: str,
        name: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
        text_format: str = "standard",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ):
        """Initialize local policy with PolicyRunner."""
        # Initialize base policy
        policy_name = name or f"local:{model_name}"
        super().__init__(
            name=policy_name,
            model_id=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Create underlying PolicyRunner
        self.runner = PolicyRunner(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            user_message_template=user_message_template,
            text_format=text_format,
        )

        # Store parameters
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

    def _compute_log_prob_impl(self, context: str, response: str) -> float:
        """
        Compute log probability using PolicyRunner.

        This is called by the base class compute_log_prob method.
        """
        # PolicyRunner.log_prob returns a raw float
        # PolicyRunner.log_prob might return a tuple, but we only need the float
        result = self.runner.log_prob(context, response)
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)

    def generate_with_logp(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Tuple[str, float, Any]]:
        """Generate responses with log probabilities."""
        # Delegate to underlying runner
        # Ensure the return type matches our signature
        results = self.runner.generate_with_logp(prompts, **kwargs)
        # Convert results to expected format
        formatted_results: List[Tuple[str, float, Any]] = []
        for result in results:
            if isinstance(result, tuple) and len(result) >= 2:
                text = str(result[0])
                logp = float(result[1])
                extra = result[2] if len(result) > 2 else None
                formatted_results.append((text, logp, extra))
            else:
                raise ValueError(f"Unexpected result format: {result}")
        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this policy."""
        stats = super().get_stats()
        stats.update(
            {
                "device": getattr(self.runner.model, "device", "unknown"),
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "top_p": self.top_p,
            }
        )
        return stats
