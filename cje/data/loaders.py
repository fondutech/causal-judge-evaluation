"""Data loading utilities following SOLID principles.

This module separates data loading concerns from the Dataset model,
following the Single Responsibility Principle.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from .models import Dataset, Sample
from .fresh_draws import FreshDrawSample, FreshDrawDataset

logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Protocol for data sources."""

    def load(self) -> List[Dict[str, Any]]:
        """Load raw data as list of dictionaries."""
        ...


class JsonlDataSource:
    """Load data from JSONL files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data


class InMemoryDataSource:
    """Load data from in-memory list."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def load(self) -> List[Dict[str, Any]]:
        """Return the in-memory data."""
        return self.data


class DatasetLoader:
    """Loads and converts raw data into typed Dataset objects.

    Follows Single Responsibility Principle - only handles data loading and conversion.
    """

    def __init__(
        self,
        base_policy_field: str = "base_policy_logprob",
        target_policy_logprobs_field: str = "target_policy_logprobs",
        prompt_field: str = "prompt",
        response_field: str = "response",
        reward_field: str = "reward",
    ):
        self.base_policy_field = base_policy_field
        self.target_policy_logprobs_field = target_policy_logprobs_field
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.reward_field = reward_field

    def load_from_source(
        self, source: DataSource, target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Load Dataset from a data source.

        Args:
            source: Data source to load from
            target_policies: List of target policy names. If None, auto-detected.

        Returns:
            Dataset instance
        """
        data = source.load()
        return self._convert_raw_data(data, target_policies)

    def _convert_raw_data(
        self, data: List[Dict[str, Any]], target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Convert raw data to Dataset."""
        # Auto-detect target policies if needed
        if target_policies is None:
            target_policies = self._detect_target_policies(data)

        # Convert raw data to samples
        samples = []
        for record in data:
            try:
                sample = self._convert_record_to_sample(record)
                samples.append(sample)
            except (KeyError, ValueError) as e:
                # Skip invalid records
                print(f"Skipping invalid record: {e}")
                continue

        if not samples:
            raise ValueError("No valid samples could be created from data")

        return Dataset(
            samples=samples,
            target_policies=target_policies,
            metadata={
                "source": "loader",
                "base_policy_field": self.base_policy_field,
                "target_policy_logprobs_field": self.target_policy_logprobs_field,
            },
        )

    def _detect_target_policies(self, data: List[Dict[str, Any]]) -> List[str]:
        """Auto-detect target policies from data."""
        policies = set()
        for record in data:
            if self.target_policy_logprobs_field in record:
                policies.update(record[self.target_policy_logprobs_field].keys())
        return sorted(list(policies))

    def _convert_record_to_sample(self, record: Dict[str, Any]) -> Sample:
        """Convert a single record to a Sample."""
        # Get prompt_id - required top-level field
        prompt_id = record.get("prompt_id")
        if prompt_id is None:
            raise ValueError("Record missing required 'prompt_id' field")

        # Extract reward if present (handle nested format)
        reward = None
        if self.reward_field in record:
            reward = record[self.reward_field]
            if isinstance(reward, dict):
                reward = reward.get("mean", reward.get("value"))
            if reward is not None:
                reward = float(reward)

        # Get base log prob
        base_logprob = record.get(self.base_policy_field)

        # Get target log probs
        target_logprobs = record.get(self.target_policy_logprobs_field, {})

        # Collect all other fields into metadata
        metadata = record.get("metadata", {})

        # Add any fields that aren't core fields to metadata
        core_fields = {
            "prompt_id",
            self.prompt_field,
            self.response_field,
            self.reward_field,
            self.base_policy_field,
            self.target_policy_logprobs_field,
            "metadata",
        }

        for key, value in record.items():
            if key not in core_fields:
                metadata[key] = value

        # Create Sample object
        return Sample(
            prompt_id=prompt_id,
            prompt=record[self.prompt_field],
            response=record[self.response_field],
            reward=reward,
            base_policy_logprob=base_logprob,
            target_policy_logprobs=target_logprobs,
            metadata=metadata,
        )


class FreshDrawLoader:
    """Loader for fresh draw samples used in DR estimation."""

    @staticmethod
    def load_from_jsonl(path: str) -> Dict[str, FreshDrawDataset]:
        """Load fresh draws from JSONL file, grouped by policy.

        Expected JSONL format:
        {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.85, "draw_idx": 0}
        {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.82, "draw_idx": 1}
        {"prompt_id": "1", "target_policy": "premium", "judge_score": 0.90, "draw_idx": 0}

        Args:
            path: Path to JSONL file containing fresh draws

        Returns:
            Dict mapping policy names to FreshDrawDataset objects
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Fresh draws file not found: {path_obj}")

        # Group samples by policy
        samples_by_policy: Dict[str, List[FreshDrawSample]] = defaultdict(list)

        with open(path_obj, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    # Create FreshDrawSample
                    sample = FreshDrawSample(
                        prompt_id=data["prompt_id"],
                        target_policy=data["target_policy"],
                        judge_score=data["judge_score"],
                        response=data.get("response"),  # Optional
                        draw_idx=data.get(
                            "draw_idx", 0
                        ),  # Default to 0 if not provided
                        fold_id=data.get("fold_id"),  # Optional
                    )

                    samples_by_policy[sample.target_policy].append(sample)

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid line {line_num}: {e}")

        # Create FreshDrawDataset for each policy
        datasets = {}
        for policy, samples in samples_by_policy.items():
            # Determine draws_per_prompt
            prompt_counts: Dict[str, int] = defaultdict(int)
            for sample in samples:
                prompt_counts[sample.prompt_id] += 1

            # Check consistency
            draws_counts = list(prompt_counts.values())
            if draws_counts and len(set(draws_counts)) > 1:
                logger.warning(
                    f"Inconsistent draws per prompt for {policy}: {set(draws_counts)}"
                )

            draws_per_prompt = max(draws_counts) if draws_counts else 1

            datasets[policy] = FreshDrawDataset(
                samples=samples,
                target_policy=policy,
                draws_per_prompt=draws_per_prompt,
            )

        return datasets
