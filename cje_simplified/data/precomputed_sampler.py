"""Load precomputed data for CJE estimation."""

import json
import math
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import numpy as np


class PrecomputedSampler:
    """Load and manage precomputed log probabilities and rewards.

    Expected data format (JSONL):
    {
        "prompt": "Input text",
        "response": "Generated response",
        "reward": 0.85,  # Calibrated reward (not raw judge score!)
        "total_logprob": -35.704,  # Base policy log P(response)
        "target_logps": {
            "pi_cot": -40.123,
            "pi_bigger": -32.456
        }
    }

    Failed log probs should be stored as null, not fallback values.

    Note: The 'reward' field should contain calibrated rewards that align
    with your business KPI. Use create_calibrated_rewards() to convert
    raw judge scores to calibrated rewards before creating the sampler.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        target_policies: Optional[List[str]] = None,
        base_policy_field: str = "p0_logprob",
        target_logps_field: str = "target_logps",
        prompt_field: str = "prompt",
        response_field: str = "response",
        reward_field: str = "reward",
    ):
        """Initialize sampler with precomputed data.

        Args:
            data: List of dictionaries with precomputed data
            target_policies: List of target policy names. If None, auto-detected.
            base_policy_field: Field name for base policy log prob
            target_logps_field: Field name for target policy log probs dict
            prompt_field: Field name for prompt/context
            response_field: Field name for response
            reward_field: Field name for calibrated reward
        """
        self.data = data
        self.base_policy_field = base_policy_field
        self.target_logps_field = target_logps_field
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.reward_field = reward_field

        # Auto-detect target policies if not provided
        if target_policies is None:
            self.target_policies = self._detect_target_policies()
        else:
            self.target_policies = target_policies

        # Validate data
        self._validate_data()

        # Prepare formatted data for estimators
        self.formatted_data = self._format_for_estimators()

    @classmethod
    def from_jsonl(
        cls, file_path: str, target_policies: Optional[List[str]] = None, **kwargs
    ) -> "PrecomputedSampler":
        """Load from JSONL file.

        Args:
            file_path: Path to JSONL file
            target_policies: Optional list of target policy names
            **kwargs: Additional arguments for __init__

        Returns:
            PrecomputedSampler instance
        """
        data = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        return cls(data, target_policies, **kwargs)

    def _detect_target_policies(self) -> List[str]:
        """Auto-detect target policies from data."""
        policies = set()
        for record in self.data:
            if self.target_logps_field in record:
                policies.update(record[self.target_logps_field].keys())
        return sorted(list(policies))

    def _validate_data(self):
        """Validate data has required fields."""
        if not self.data:
            raise ValueError("No data provided")

        # Check first record has required fields
        sample = self.data[0]
        required_fields = [
            self.prompt_field,
            self.response_field,
            self.reward_field,
            self.base_policy_field,
            self.target_logps_field,
        ]

        missing = [f for f in required_fields if f not in sample]
        if missing:
            raise ValueError(f"Missing required fields in data: {missing}")

        # Validate target policies exist
        target_logps = sample.get(self.target_logps_field, {})
        missing_policies = set(self.target_policies) - set(target_logps.keys())
        if missing_policies:
            raise ValueError(
                f"Target policies {missing_policies} not found in {self.target_logps_field}"
            )

    def _format_for_estimators(self) -> List[Dict[str, Any]]:
        """Format data for CJE estimators.

        Returns list of dicts with:
        - context: prompt text
        - response: generated text
        - logp: base policy log prob
        - reward: calibrated reward
        - logp_target_all: dict of target log probs
        """
        formatted = []

        for record in self.data:
            # Extract reward
            reward = record[self.reward_field]

            # Handle nested reward format (backwards compatibility)
            if isinstance(reward, dict):
                reward = reward.get("mean", reward.get("value"))

            reward = float(reward)

            # Skip if base log prob is invalid
            base_logp = record.get(self.base_policy_field)
            if base_logp is None:
                continue

            # Extract target log probs
            target_logps = record.get(self.target_logps_field, {})

            # Check all required target policies have valid log probs
            valid_targets = {}
            skip_record = False
            for policy in self.target_policies:
                logp = target_logps.get(policy)
                if logp is None:
                    skip_record = True
                    break
                valid_targets[policy] = logp

            if skip_record:
                continue

            formatted.append(
                {
                    "context": record[self.prompt_field],
                    "response": record[self.response_field],
                    "logp": base_logp,
                    "reward": reward,
                    "logp_target_all": valid_targets,
                }
            )

        if not formatted:
            raise ValueError("No valid records after filtering invalid log probs")

        return formatted

    def get_data_for_policy(self, target_policy: str) -> Optional[List[Dict[str, Any]]]:
        """Get formatted data for a specific target policy.

        Returns data in format expected by estimators:
        - reward: float
        - total_logprob: base policy log prob
        - policy_logprob: target policy log prob
        """
        if target_policy not in self.target_policies:
            return None

        policy_data = []
        for record in self.data:
            # Check if we have valid log probs
            base_logp = record.get(self.base_policy_field)
            if base_logp is None:
                continue

            target_logps = record.get(self.target_logps_field, {})
            target_logp = target_logps.get(target_policy)
            if target_logp is None:
                continue

            # Extract reward
            reward = record[self.reward_field]
            if isinstance(reward, dict):
                reward = reward.get("mean", reward.get("value"))

            policy_data.append(
                {
                    "reward": float(reward),
                    "total_logprob": base_logp,
                    "policy_logprob": target_logp,
                    "prompt": record.get(self.prompt_field, ""),
                    "response": record.get(self.response_field, ""),
                }
            )

        return policy_data if policy_data else None

    def compute_importance_weights(
        self, target_policy: str, clip_weight: float = 100.0
    ) -> np.ndarray:
        """Compute importance weights for a target policy.

        Args:
            target_policy: Name of target policy
            clip_weight: Maximum weight value (for variance control)

        Returns:
            Array of importance weights
        """
        if target_policy not in self.target_policies:
            raise ValueError(f"Unknown target policy: {target_policy}")

        weights = []
        for record in self.formatted_data:
            base_logp = record["logp"]
            target_logp = record["logp_target_all"][target_policy]

            # Compute weight with overflow protection
            log_ratio = target_logp - base_logp
            if log_ratio > 50:  # exp(50) is huge
                weight = clip_weight
            elif log_ratio < -50:  # exp(-50) is tiny
                weight = 0.0
            else:
                weight = min(math.exp(log_ratio), clip_weight)

            weights.append(weight)

        return np.array(weights)

    def get_rewards(self) -> np.ndarray:
        """Get array of calibrated rewards."""
        return np.array([r["reward"] for r in self.formatted_data])

    def get_contexts(self) -> List[str]:
        """Get list of contexts/prompts."""
        return [r["context"] for r in self.formatted_data]

    def get_responses(self) -> List[str]:
        """Get list of responses."""
        return [r["response"] for r in self.formatted_data]

    @property
    def n_samples(self) -> int:
        """Number of valid samples."""
        return len(self.formatted_data)

    @property
    def n_policies(self) -> int:
        """Number of target policies."""
        return len(self.target_policies)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "n_samples": self.n_samples,
            "n_policies": self.n_policies,
            "target_policies": self.target_policies,
            "reward_mean": float(np.mean(self.get_rewards())),
            "reward_std": float(np.std(self.get_rewards())),
            "n_invalid_dropped": len(self.data) - self.n_samples,
        }
