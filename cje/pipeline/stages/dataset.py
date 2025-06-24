"""
Dataset loading stage - Handles dataset loading and validation.
"""

import logging
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from itertools import islice

from rich.console import Console

from ...data import load_dataset
from ...data.base import CJEDataset
from ...data.validation import validate_dataset
from ...cache import compute_contexts_hash, chunk_exists, load_chunk, save_chunk
from ..validation import validate_stage_output

logger = logging.getLogger(__name__)


class DatasetStage:
    """Handles dataset loading, validation, and caching."""

    def __init__(self, work_dir: Path, console: Optional[Console] = None):
        self.work_dir = work_dir
        self.console = console or Console()

    @validate_stage_output(
        required_fields={"uid", "context"},
        optional_fields={"response", "reward", "y_true", "meta"},
    )
    def run(
        self,
        dataset_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Load and prepare dataset for the pipeline.

        Args:
            dataset_config: Dataset configuration

        Returns:
            List of dataset rows as dictionaries
        """
        dataset_name = dataset_config["name"]
        split = dataset_config.get("split", "train")
        sample_limit = dataset_config.get("sample_limit")

        logger.info(f"Loading dataset: {dataset_name}, split: {split}")
        self.console.print(f"[bold blue]📊 Loading dataset: {dataset_name}[/bold blue]")

        # Load dataset
        with self.console.status("[bold blue]Loading dataset..."):
            ds = load_dataset(dataset_name, split=split)

        # Get dataset size if possible
        try:
            if hasattr(ds, "__len__"):
                dataset_size = len(ds)
                logger.info(f"Dataset loaded: {dataset_size} samples")
                self.console.print(
                    f"[green]✅ Dataset loaded: {dataset_size} samples[/green]"
                )
            else:
                logger.info("Dataset loaded (size unknown)")
                self.console.print("[green]✅ Dataset loaded[/green]")
        except:
            logger.info("Dataset loaded (size unknown)")
            self.console.print("[green]✅ Dataset loaded[/green]")

        # Apply sample limit if specified
        if sample_limit is not None:
            logger.info(f"Limiting to first {sample_limit} samples")
            self.console.print(
                f"[yellow]⚠️  Limiting to first {sample_limit} samples for testing[/yellow]"
            )
            rows = self._apply_sample_limit(ds, sample_limit)
        else:
            rows = self._dataset_to_rows(ds)

        # Validate dataset
        self._validate_dataset(rows)

        # Compute and cache contexts
        contexts_hash = self._cache_contexts(rows)

        return rows

    def _apply_sample_limit(
        self, ds: CJEDataset, sample_limit: int
    ) -> List[Dict[str, Any]]:
        """Apply sample limit to dataset."""
        with self.console.status("[bold blue]Sampling limited dataset..."):
            # Check if we can optimize iteration
            # Collect samples up to the limit
            samples = []
            sample_count = 0

            # Use iterator to get samples
            for sample in ds.itersamples():
                if sample_count >= sample_limit:
                    break
                # Convert sample to dict - much simpler now!
                sample_dict: Dict[str, Any] = {
                    "uid": sample.uid,
                    "context": sample.context,
                }
                # Add optional fields if present
                if sample.response is not None:
                    sample_dict["response"] = sample.response
                if sample.y_true is not None:
                    sample_dict["y_true"] = sample.y_true
                if sample.logp is not None:
                    sample_dict["logp"] = sample.logp
                if sample.reward is not None:
                    sample_dict["reward"] = sample.reward
                if sample.meta:
                    sample_dict["meta"] = sample.meta

                samples.append(sample_dict)
                sample_count += 1

        return samples

    def _dataset_to_rows(self, ds: CJEDataset) -> List[Dict[str, Any]]:
        """Convert dataset to list of dictionaries."""
        rows = []

        with self.console.status("[bold blue]Loading dataset samples..."):
            for sample in ds.itersamples():
                # Convert sample to dict - much simpler now!
                sample_dict: Dict[str, Any] = {
                    "uid": sample.uid,
                    "context": sample.context,
                }
                # Add optional fields if present
                if sample.response is not None:
                    sample_dict["response"] = sample.response
                if sample.y_true is not None:
                    sample_dict["y_true"] = sample.y_true
                if sample.logp is not None:
                    sample_dict["logp"] = sample.logp
                if sample.reward is not None:
                    sample_dict["reward"] = sample.reward
                if sample.meta:
                    sample_dict["meta"] = sample.meta

                rows.append(sample_dict)

        return rows

    def _validate_dataset(self, rows: List[Dict[str, Any]]) -> None:
        """Validate dataset structure and content."""
        if not rows:
            raise ValueError("Dataset is empty")

        # Basic validation
        required_fields = {"uid", "context"}
        first_row = rows[0]
        missing_fields = required_fields - set(first_row.keys())

        if missing_fields:
            raise ValueError(f"Dataset missing required fields: {missing_fields}")

        logger.info(f"Dataset validated: {len(rows)} rows")

    def _cache_contexts(self, rows: List[Dict[str, Any]]) -> str:
        """Cache contexts and return hash."""
        # Compute contexts hash based on dataset config only
        # (we don't have logging policy config at this stage)
        import hashlib
        import json

        hash_data = {"dataset": self._get_dataset_config_for_hash(rows)}
        contexts_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Save contexts to cache
        contexts = [row["context"] for row in rows]
        if not chunk_exists(self.work_dir, "contexts", contexts_hash):
            save_chunk(
                self.work_dir,
                "contexts",
                contexts_hash,
                contexts,
                metadata={"count": len(contexts)},
            )
            logger.info(f"Contexts cached: {contexts_hash}")

        return contexts_hash

    def _get_dataset_config_for_hash(
        self, rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get dataset configuration for hashing."""
        return {
            "num_samples": len(rows),
            "has_contexts": all("context" in row for row in rows),
        }
