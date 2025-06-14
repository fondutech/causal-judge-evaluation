"""
Unified data loader for CJE - handles all data formats simply.
"""

from typing import List, Dict, Any, Optional, Iterator, Union
from pathlib import Path
import json
import csv
from dataclasses import dataclass
import pandas as pd
from abc import ABC, abstractmethod


@dataclass
class DataSample:
    """Single data sample with all required fields."""

    context: str
    response: str
    reward: Optional[float] = None
    logp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        d: Dict[str, Any] = {
            "context": self.context,
            "response": self.response,
        }
        if self.reward is not None:
            d["reward"] = self.reward
        if self.logp is not None:
            d["logp"] = self.logp
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class DataLoader:
    """
    Unified data loader that handles multiple formats.

    Supports:
    - JSONL files
    - CSV files
    - JSON arrays
    - ChatBot Arena format
    - In-memory lists
    """

    @staticmethod
    def load(
        source: Union[str, Path, List[Dict[str, Any]]],
        format: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[DataSample]:
        """
        Load data from various sources.

        Args:
            source: File path, dataset name, or list of dicts
            format: Optional format hint ('jsonl', 'csv', 'arena', etc.)
            limit: Maximum number of samples to load
            **kwargs: Additional format-specific options

        Returns:
            List of DataSample objects
        """
        # Handle in-memory data
        if isinstance(source, list):
            samples = [DataLoader._dict_to_sample(d) for d in source]
            return samples[:limit] if limit else samples

        # Convert to Path
        if isinstance(source, str):
            # Check if it's a known dataset name
            if source.lower() == "chatbotarena":
                return DataLoader._load_arena(limit=limit, **kwargs)
            source = Path(source)

        # Auto-detect format from extension if not specified
        if format is None:
            if source.suffix == ".jsonl":
                format = "jsonl"
            elif source.suffix == ".csv":
                format = "csv"
            elif source.suffix == ".json":
                format = "json"
            else:
                # Try to guess from content
                format = DataLoader._guess_format(source)

        # Load based on format
        if format == "jsonl":
            return DataLoader._load_jsonl(source, limit=limit, **kwargs)
        elif format == "csv":
            return DataLoader._load_csv(source, limit=limit, **kwargs)
        elif format == "json":
            return DataLoader._load_json(source, limit=limit, **kwargs)
        elif format == "arena":
            return DataLoader._load_arena(limit=limit, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def _dict_to_sample(d: Dict[str, Any]) -> DataSample:
        """Convert dictionary to DataSample."""
        # Handle various field names
        context = d.get("context") or d.get("prompt") or d.get("question") or ""
        response = d.get("response") or d.get("answer") or d.get("completion") or ""
        reward = d.get("reward") or d.get("score") or d.get("rating")
        logp = d.get("logp") or d.get("log_prob") or d.get("logprob")

        # Everything else goes to metadata
        metadata = {
            k: v
            for k, v in d.items()
            if k
            not in [
                "context",
                "prompt",
                "question",
                "response",
                "answer",
                "completion",
                "reward",
                "score",
                "rating",
                "logp",
                "log_prob",
                "logprob",
            ]
        }

        return DataSample(
            context=str(context),
            response=str(response),
            reward=float(reward) if reward is not None else None,
            logp=float(logp) if logp is not None else None,
            metadata=metadata if metadata else None,
        )

    @staticmethod
    def _load_jsonl(
        path: Path, limit: Optional[int] = None, **kwargs: Any
    ) -> List[DataSample]:
        """Load from JSONL file."""
        samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data = json.loads(line.strip())
                samples.append(DataLoader._dict_to_sample(data))
        return samples

    @staticmethod
    def _load_csv(
        path: Path,
        limit: Optional[int] = None,
        context_col: str = "context",
        response_col: str = "response",
        reward_col: Optional[str] = "reward",
        **kwargs: Any,
    ) -> List[DataSample]:
        """Load from CSV file."""
        df = pd.read_csv(path, nrows=limit)

        samples = []
        for _, row in df.iterrows():
            sample = DataSample(
                context=str(row[context_col]),
                response=str(row[response_col]),
                reward=(
                    float(row[reward_col]) if reward_col and reward_col in row else None
                ),
                metadata=row.to_dict(),
            )
            samples.append(sample)

        return samples

    @staticmethod
    def _load_json(
        path: Path, limit: Optional[int] = None, **kwargs: Any
    ) -> List[DataSample]:
        """Load from JSON array file."""
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of samples")

        samples = [DataLoader._dict_to_sample(d) for d in data]
        return samples[:limit] if limit else samples

    @staticmethod
    def _load_arena(
        limit: Optional[int] = None, split: str = "train", **kwargs: Any
    ) -> List[DataSample]:
        """Load ChatBot Arena dataset."""
        # This is a simplified version - in practice would download/cache
        from datasets import load_dataset

        dataset = load_dataset("lmsys/chatbot_arena_conversations", split=split)

        samples = []
        for i, item in enumerate(dataset):
            if limit and i >= limit:
                break

            # Extract conversation
            conv = item.get("conversation_a", [])
            if len(conv) >= 2:
                context = conv[0].get("content", "")
                response = conv[1].get("content", "")
            else:
                continue

            # Get winner as reward (simplified)
            winner = item.get("winner", "tie")
            reward = 1.0 if winner == "model_a" else 0.0 if winner == "model_b" else 0.5

            samples.append(
                DataSample(
                    context=context,
                    response=response,
                    reward=reward,
                    metadata={
                        "model_a": item.get("model_a"),
                        "model_b": item.get("model_b"),
                        "winner": winner,
                    },
                )
            )

        return samples

    @staticmethod
    def _guess_format(path: Path) -> str:
        """Try to guess format from file content."""
        with open(path, "r") as f:
            first_line = f.readline().strip()

        # Check if it's JSON
        try:
            json.loads(first_line)
            return "jsonl"
        except:
            pass

        # Check if it's CSV (has commas and possibly headers)
        if "," in first_line:
            return "csv"

        # Default to JSONL
        return "jsonl"


class DataIterator:
    """
    Memory-efficient iterator for large datasets.
    """

    def __init__(
        self,
        source: Union[str, Path],
        format: Optional[str] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ):
        self.source = Path(source) if isinstance(source, str) else source
        self.format = format
        self.batch_size = batch_size
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over samples one at a time."""
        if self.format == "jsonl" or self.source.suffix == ".jsonl":
            with open(self.source) as f:
                for line in f:
                    data = json.loads(line.strip())
                    yield DataLoader._dict_to_sample(data)
        else:
            # For other formats, load in batches
            offset = 0
            while True:
                batch = DataLoader.load(
                    self.source,
                    format=self.format,
                    limit=self.batch_size,
                    offset=offset,
                    **self.kwargs,
                )
                if not batch:
                    break

                for sample in batch:
                    yield sample

                if len(batch) < self.batch_size:
                    break

                offset += self.batch_size


# Convenience functions
def load_data(
    source: Union[str, Path, List[Dict[str, Any]]], **kwargs: Any
) -> List[DataSample]:
    """Convenience function to load data."""
    return DataLoader.load(source, **kwargs)


def iter_data(source: Union[str, Path], **kwargs: Any) -> DataIterator:
    """Convenience function to create data iterator."""
    return DataIterator(source, **kwargs)
