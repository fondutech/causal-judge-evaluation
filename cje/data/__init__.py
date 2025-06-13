"""CJE dataset loading and processing utilities."""

from enum import Enum
from typing import Literal, Union, Optional, Type, Dict
from pathlib import Path
import logging

from .base import CJEDataset
from .schema import CJESample
from .jsonl import JSONLDataset
from .csv_dataset import CSVDataset
from .pairwise import PairwiseComparisonDataset, BradleyTerryModel
from .chatbot_arena import ChatbotArenaDataset
from .validation import (
    validate_dataset,
    validate_input_data,
    check_propensity_quality,
    check_ground_truth_quality,
    ValidationResult,
)
from .trajectory_dataset import TrajectoryJSONLDataset

logger = logging.getLogger(__name__)


class DatasetName(str, Enum):
    """Available built-in datasets."""

    PAIRWISE = "PairwiseComparison"
    CHATBOT_ARENA = "ChatbotArena"
    TRAJECTORY_JSONL = "TrajectoryJSONL"


_DATASETS: Dict[DatasetName, Type[CJEDataset]] = {
    DatasetName.PAIRWISE: PairwiseComparisonDataset,
    DatasetName.CHATBOT_ARENA: ChatbotArenaDataset,
    DatasetName.TRAJECTORY_JSONL: TrajectoryJSONLDataset,
}


def load_dataset(
    name: Union[DatasetName, str, Path], split: Optional[str] = None
) -> CJEDataset:
    """Load a dataset by name or file path.

    Args:
        name: Dataset to load. Can be:
            - Built-in dataset name (e.g., "ChatbotArena", "PairwiseComparison")
            - Path to data file (e.g., "/path/to/data.jsonl", "/path/to/data.csv")
            - DatasetName enum instance
        split: Dataset split to load. If None, uses dataset-specific default behavior.
               Not applicable for data files.

    Returns:
        An instance of the requested dataset

    Raises:
        FileNotFoundError: If a file path is provided but the file doesn't exist
        ValueError: If the dataset name/path is not recognized or invalid

    Examples:
        # Load built-in dataset
        ds = load_dataset("ChatbotArena", split="train")

        # Load custom data files
        ds = load_dataset("/path/to/my_data.jsonl")
        ds = load_dataset("./experiments/data.csv")
        ds = load_dataset("./results/results.tsv")
    """
    # Check if it's a file path (string or Path object)
    if isinstance(name, (str, Path)):
        path = Path(name)

        # If it's an existing file with appropriate extension, load accordingly
        if path.exists():
            # Handle JSON/JSONL files
            if path.suffix.lower() in [".jsonl", ".json"]:
                if split is not None:
                    # Log a warning but don't fail - split is ignored for file paths
                    logger.warning(f"Split '{split}' ignored for JSONL file: {path}")
                return JSONLDataset(str(path))

            # Handle CSV/TSV files
            elif path.suffix.lower() in [".csv", ".tsv"]:
                if split is not None:
                    # Log a warning but don't fail - split is ignored for file paths
                    logger.warning(f"Split '{split}' ignored for CSV file: {path}")
                return CSVDataset(str(path))

            else:
                # File exists but doesn't have supported extension
                raise ValueError(
                    f"Unsupported file type: {path.suffix}. "
                    f"Supported extensions: .jsonl, .json, .csv, .tsv"
                )

        # If it doesn't exist as a file, try as built-in dataset name
        if not path.exists():
            # Try to interpret as built-in dataset name
            try:
                dataset_name_enum = DatasetName(str(name))
            except ValueError as e:
                # Check if it looks like a file path that doesn't exist
                if (
                    str(name).endswith((".jsonl", ".json", ".csv", ".tsv"))
                    or "/" in str(name)
                    or "\\" in str(name)
                ):
                    raise FileNotFoundError(f"File not found: {name}")
                else:
                    raise ValueError(
                        f"Invalid dataset name: '{name}'. "
                        f"Valid built-in datasets: {[e.value for e in DatasetName]}. "
                        f"Or provide path to existing file (supported: .jsonl, .json, .csv, .tsv)."
                    ) from e
    elif isinstance(name, DatasetName):
        dataset_name_enum = name
    else:
        raise TypeError(
            f"Dataset name must be a DatasetName enum, string, or Path, not {type(name)}"
        )

    # Handle built-in datasets
    if dataset_name_enum not in _DATASETS:
        # This case should ideally not be reached if DatasetName enum is exhaustive
        # and _DATASETS keys are solely from DatasetName.
        raise KeyError(
            f"Dataset '{dataset_name_enum}' not found in _DATASETS registry."
        )

    # Default split behavior for built-in datasets
    if split is None:
        split = "train"

    dataset_class = _DATASETS[dataset_name_enum]
    # Type assertion: all built-in dataset classes have a download class method
    return dataset_class.download(split=split)  # type: ignore[attr-defined]


# Export public API
__all__ = [
    "CJEDataset",
    "CJESample",
    "DatasetName",
    "load_dataset",
    # Dataset classes
    "JSONLDataset",
    "CSVDataset",
    "PairwiseComparisonDataset",
    "ChatbotArenaDataset",
    "BradleyTerryModel",
    "TrajectoryJSONLDataset",
    # Validation functions
    "validate_dataset",
    "validate_input_data",
    "check_propensity_quality",
    "check_ground_truth_quality",
    "ValidationResult",
]
