"""Load precomputed data for CJE estimation."""

import json
import math
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import numpy as np
from .models import Sample, Dataset


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
        self.raw_data = data
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

        # Convert raw data to Sample objects
        samples = self._create_samples()
        
        # Create Dataset object
        self.dataset = Dataset(
            samples=samples,
            target_policies=self.target_policies,
            metadata={"source": "PrecomputedSampler"}
        )

        # Prepare formatted data for backwards compatibility
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
        for record in self.raw_data:
            if self.target_logps_field in record:
                policies.update(record[self.target_logps_field].keys())
        return sorted(list(policies))

    def _create_samples(self) -> List[Sample]:
        """Convert raw data to Sample objects."""
        samples = []
        for record in self.raw_data:
            try:
                # Extract reward (handle nested format)
                reward = record[self.reward_field]
                if isinstance(reward, dict):
                    reward = reward.get("mean", reward.get("value"))
                
                # Get base log prob (required)
                base_logprob = record.get(self.base_policy_field)
                
                # Get target log probs
                target_logprobs = record.get(self.target_logps_field, {})
                
                # Create Sample object
                sample = Sample(
                    prompt=record[self.prompt_field],
                    response=record[self.response_field],
                    reward=float(reward),
                    base_logprob=base_logprob,
                    target_logprobs=target_logprobs,
                    metadata=record.get("metadata", {})
                )
                samples.append(sample)
            except (KeyError, ValueError) as e:
                # Skip invalid records
                print(f"Skipping invalid record: {e}")
                continue
        
        if not samples:
            raise ValueError("No valid samples could be created from data")
        
        return samples

    def _validate_data(self):
        """Legacy method for backwards compatibility - validation now done by Pydantic."""
        # Validation is now handled by the Dataset model
        pass

    def _format_for_estimators(self) -> List[Dict[str, Any]]:
        """Format data for CJE estimators (backwards compatibility).

        Returns list of dicts with:
        - context: prompt text
        - response: generated text
        - logp: base policy log prob
        - reward: calibrated reward
        - logp_target_all: dict of target log probs
        """
        formatted = []

        for sample in self.dataset.samples:
            # Skip samples without valid base log prob
            if sample.base_logprob is None:
                continue

            # Check all required target policies have valid log probs
            valid_targets = {}
            skip_record = False
            for policy in self.target_policies:
                logp = sample.target_logprobs.get(policy)
                if logp is None:
                    skip_record = True
                    break
                valid_targets[policy] = logp

            if skip_record:
                continue

            formatted.append(
                {
                    "context": sample.prompt,
                    "response": sample.response,
                    "logp": sample.base_logprob,
                    "reward": sample.reward,
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

        # Use Dataset's filter method to get valid samples
        valid_samples = self.dataset.filter_valid_samples(target_policy)
        if not valid_samples:
            return None

        policy_data = []
        for sample in valid_samples:
            policy_data.append(
                {
                    "reward": sample.reward,
                    "total_logprob": sample.base_logprob,
                    "policy_logprob": sample.target_logprobs[target_policy],
                    "prompt": sample.prompt,
                    "response": sample.response,
                }
            )

        return policy_data

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
        return np.array([s.reward for s in self.dataset.samples])

    def get_contexts(self) -> List[str]:
        """Get list of contexts/prompts."""
        return [s.prompt for s in self.dataset.samples]

    def get_responses(self) -> List[str]:
        """Get list of responses."""
        return [s.response for s in self.dataset.samples]

    @property
    def n_samples(self) -> int:
        """Number of valid samples."""
        return self.dataset.n_samples

    @property
    def n_policies(self) -> int:
        """Number of target policies."""
        return len(self.target_policies)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        dataset_summary = self.dataset.summary()
        return {
            "n_samples": self.n_samples,
            "n_policies": self.n_policies,
            "target_policies": self.target_policies,
            "reward_mean": dataset_summary["reward_mean"],
            "reward_std": dataset_summary["reward_std"],
            "n_invalid_dropped": len(self.raw_data) - self.n_samples,
            "valid_samples_per_policy": dataset_summary["valid_samples_per_policy"],
        }
