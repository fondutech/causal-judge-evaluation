"""Data loading utilities following SOLID principles.

This module separates data loading concerns from the Dataset model,
following the Single Responsibility Principle.
"""

import json
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod

from .models import Dataset, Sample


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
        # Extract reward (handle nested format)
        reward = record[self.reward_field]
        if isinstance(reward, dict):
            reward = reward.get("mean", reward.get("value"))

        # Get base log prob
        base_logprob = record.get(self.base_policy_field)

        # Get target log probs
        target_logprobs = record.get(self.target_policy_logprobs_field, {})

        # Create Sample object
        return Sample(
            prompt=record[self.prompt_field],
            response=record[self.response_field],
            reward=float(reward),
            base_policy_logprob=base_logprob,
            target_policy_logprobs=target_logprobs,
            metadata=record.get("metadata", {}),
        )
