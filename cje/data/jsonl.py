"""JSONL dataset loader for custom data files."""

import json
from pathlib import Path
from typing import Iterable, Optional

from .base import CJEDataset
from .schema import CJESample


class JSONLDataset(CJEDataset):
    """Dataset loader for JSONL (JSON Lines) files.

    This allows users to load their own data files in JSONL format,
    where each line is a JSON object representing a sample.

    Example usage:
        # Load from file path
        dataset = JSONLDataset("/path/to/data.jsonl")

        # Iterate over samples
        for sample in dataset.itersamples():
            print(sample.context, sample.response)
    """

    def __init__(self, file_path: str):
        """Initialize with path to JSONL file.

        Args:
            file_path: Path to the JSONL file
        """
        self.file_path = Path(file_path)
        self.name = str(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {file_path}")
        if not self.file_path.suffix.lower() in [".jsonl", ".json"]:
            raise ValueError(f"File must have .jsonl or .json extension: {file_path}")

    @classmethod
    def download(
        cls, cache_dir: Optional[str] = None, split: str = "train"
    ) -> "JSONLDataset":
        """Not applicable for JSONL files - use constructor directly."""
        raise NotImplementedError(
            "JSONLDataset loads from local files. Use JSONLDataset(file_path) directly."
        )

    def itersamples(self) -> Iterable[CJESample]:
        """Lazily yield CJESample objects from the JSONL file.

        Yields:
            CJESample objects representing individual examples.

        Raises:
            json.JSONDecodeError: If a line contains invalid JSON
            ValueError: If required fields are missing
        """
        line_number = 0
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Invalid JSON on line {line_number}: {e.msg}", e.doc, e.pos
                        )

                    # Validate required fields
                    if not isinstance(data, dict):
                        raise ValueError(
                            f"Line {line_number}: Expected JSON object, got {type(data)}"
                        )

                    # Auto-generate uid if missing
                    if "uid" not in data:
                        data["uid"] = f"sample_{line_number}"

                    if "context" not in data:
                        raise ValueError(
                            f"Line {line_number}: Missing required field 'context'"
                        )

                    # Extract target_samples if present (for Scenario 3)
                    target_samples = data.get("target_samples")
                    meta = {}
                    if target_samples is not None:
                        meta["target_samples"] = target_samples

                    # Add other non-standard fields to meta
                    for key in [
                        "score",
                        "score_raw",
                        "score_cal",
                        "logp_target",
                        "reward",
                    ]:
                        if key in data:
                            meta[key] = data[key]

                    # Create CJESample
                    yield CJESample(
                        uid=data["uid"],
                        context=data["context"],
                        response=data.get("response", ""),
                        y_true=data.get("y_true"),
                        logp=data.get("logp"),
                        meta=meta if meta else {},
                    )

        except FileNotFoundError:
            raise FileNotFoundError(f"JSONL file not found: {self.file_path}")
        except Exception as e:
            if isinstance(e, (json.JSONDecodeError, ValueError)):
                raise
            raise RuntimeError(f"Error reading JSONL file at line {line_number}: {e}")

    def __len__(self) -> int:
        """Get number of samples in the dataset.

        Note: This requires reading through the entire file.
        """
        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    count += 1
        return count

    def __str__(self) -> str:
        return f"JSONLDataset({self.file_path})"

    def __repr__(self) -> str:
        return f"JSONLDataset(file_path='{self.file_path}')"
