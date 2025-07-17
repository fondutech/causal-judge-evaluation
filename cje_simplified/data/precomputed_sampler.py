"""Load precomputed data for CJE estimation."""

import json
import math
from typing import List, Dict, Any, Optional, Set, Union
from pathlib import Path
import numpy as np
from .models import Sample, Dataset


class PrecomputedSampler:
    """Adapter that provides CJE-specific operations on a Dataset.

    This class wraps a Dataset to provide CJE-specific functionality
    like importance weight computation.

    For new code, prefer using Dataset directly and its class methods:
    - Dataset.from_raw_data()
    - Dataset.from_jsonl()
    """

    def __init__(
        self,
        data_or_dataset: Union[Dataset, List[Dict[str, Any]]],
        target_policies: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize sampler.

        Args:
            data_or_dataset: Either a Dataset instance or raw data list
            target_policies: Target policy names (only used if data_or_dataset is a list)
            **kwargs: Additional arguments passed to Dataset.from_raw_data()
        """
        if isinstance(data_or_dataset, Dataset):
            self.dataset = data_or_dataset
        else:
            # Create Dataset from raw data
            self.dataset = Dataset.from_raw_data(
                data_or_dataset, target_policies=target_policies, **kwargs
            )

        self.target_policies = self.dataset.target_policies

        # Prepare formatted data
        self.formatted_data = self._format_for_estimators()

    @classmethod
    def from_jsonl(
        cls, file_path: str, target_policies: Optional[List[str]] = None, **kwargs: Any
    ) -> "PrecomputedSampler":
        """Load from JSONL file.

        Args:
            file_path: Path to JSONL file
            target_policies: Optional list of target policy names
            **kwargs: Additional arguments for Dataset.from_raw_data

        Returns:
            PrecomputedSampler instance
        """
        dataset = Dataset.from_jsonl(file_path, target_policies, **kwargs)
        return cls(dataset)

    def _format_for_estimators(self) -> List[Dict[str, Any]]:
        """Format data for CJE estimators.

        Returns list of dicts with:
        - context: prompt text
        - response: generated text
        - base_policy_logprob: base policy log prob
        - reward: calibrated reward
        - target_policy_logprobs: dict of target log probs
        """
        formatted = []

        for sample in self.dataset.samples:
            # Skip samples without valid base log prob
            if sample.base_policy_logprob is None:
                continue

            # Check all required target policies have valid log probs
            valid_targets = {}
            skip_record = False
            for policy in self.target_policies:
                logp = sample.target_policy_logprobs.get(policy)
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
                    "base_policy_logprob": sample.base_policy_logprob,
                    "reward": sample.reward,
                    "target_policy_logprobs": valid_targets,
                }
            )

        if not formatted:
            raise ValueError("No valid records after filtering invalid log probs")

        return formatted

    def get_data_for_policy(self, target_policy: str) -> Optional[List[Dict[str, Any]]]:
        """Get formatted data for a specific target policy.

        Returns data in format expected by estimators:
        - reward: float
        - base_policy_logprob: base policy log prob
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
                    "base_policy_logprob": sample.base_policy_logprob,
                    "policy_logprob": sample.target_policy_logprobs[target_policy],
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
            base_logp = record["base_policy_logprob"]
            target_logp = record["target_policy_logprobs"][target_policy]

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
            "n_invalid_dropped": 0,  # No longer tracked
            "valid_samples_per_policy": dataset_summary["valid_samples_per_policy"],
        }
