"""
Checkpointing utilities for resumable operations in CJE.

This module provides generic checkpointing capabilities that can be used
for any long-running operation that processes items with unique identifiers.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypeVar, Generic, Callable
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Type of items being processed


class CheckpointManager(Generic[T]):
    """Generic checkpoint manager for resumable operations."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        get_uid_fn: Callable[[T], str] = lambda x: (
            x.get("uid", str(x)) if isinstance(x, dict) else str(x)
        ),
        serialize_fn: Callable[[T], Dict[str, Any]] = lambda x: (
            x if isinstance(x, dict) else x.__dict__
        ),
        deserialize_fn: Callable[[Dict[str, Any]], T] = lambda x: x,  # type: ignore
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_path: Path to checkpoint file (JSONL format)
            get_uid_fn: Function to extract unique ID from an item
            serialize_fn: Function to serialize item to dict for JSON storage
            deserialize_fn: Function to deserialize dict back to item
        """
        self.checkpoint_path = checkpoint_path
        self.get_uid_fn = get_uid_fn
        self.serialize_fn = serialize_fn
        self.deserialize_fn = deserialize_fn
        self._processed_items: List[T] = []
        self._processed_uids: Set[str] = set()
        self._loaded_from_checkpoint = False

    def load_checkpoint(self) -> List[T]:
        """Load existing checkpoint if it exists."""
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return []

        try:
            items = []
            with open(self.checkpoint_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        item = self.deserialize_fn(data)
                        items.append(item)

            self._processed_items = items
            self._processed_uids = {self.get_uid_fn(item) for item in items}
            self._loaded_from_checkpoint = True

            logger.info(
                f"Loaded checkpoint with {len(items):,} existing items from {self.checkpoint_path}"
            )
            return items

        except Exception as e:
            logger.warning(
                f"Could not load checkpoint from {self.checkpoint_path}: {e}"
            )
            return []

    def save_checkpoint(self, items: Optional[List[T]] = None) -> None:
        """Save current progress to checkpoint file."""
        if not self.checkpoint_path:
            return

        items_to_save = items if items is not None else self._processed_items

        try:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_path, "w") as f:
                for item in items_to_save:
                    serialized = self.serialize_fn(item)
                    f.write(json.dumps(serialized) + "\n")

            logger.debug(
                f"Saved checkpoint with {len(items_to_save):,} items to {self.checkpoint_path}"
            )

        except Exception as e:
            logger.warning(f"Could not save checkpoint to {self.checkpoint_path}: {e}")

    def append_to_checkpoint(self, items: List[T]) -> None:
        """Append new items to checkpoint file."""
        if not self.checkpoint_path or not items:
            return

        try:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_path, "a") as f:
                for item in items:
                    serialized = self.serialize_fn(item)
                    f.write(json.dumps(serialized) + "\n")

            logger.debug(
                f"Appended {len(items):,} items to checkpoint {self.checkpoint_path}"
            )

        except Exception as e:
            logger.warning(
                f"Could not append to checkpoint {self.checkpoint_path}: {e}"
            )

    def is_processed(self, item: T) -> bool:
        """Check if an item has already been processed."""
        uid = self.get_uid_fn(item)
        return uid in self._processed_uids

    def mark_processed(self, item: T) -> None:
        """Mark an item as processed."""
        uid = self.get_uid_fn(item)
        if uid not in self._processed_uids:
            self._processed_items.append(item)
            self._processed_uids.add(uid)

    def filter_unprocessed(self, items: List[T]) -> List[T]:
        """Filter out items that have already been processed."""
        return [item for item in items if not self.is_processed(item)]

    def get_processed_items(self) -> List[T]:
        """Get all processed items."""
        return self._processed_items.copy()

    def get_processed_count(self) -> int:
        """Get count of processed items."""
        return len(self._processed_items)

    def was_loaded_from_checkpoint(self) -> bool:
        """Check if data was loaded from an existing checkpoint."""
        return self._loaded_from_checkpoint


class BatchProcessor(Generic[T]):
    """Generic batch processor with checkpointing and progress tracking."""

    def __init__(
        self,
        batch_size: int = 16,
        checkpoint_manager: Optional[CheckpointManager[T]] = None,
        continue_on_error: bool = True,
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items to process in each batch
            checkpoint_manager: Optional checkpoint manager for resumable processing
            continue_on_error: Whether to continue processing if a batch fails
        """
        self.batch_size = batch_size
        self.checkpoint_manager = checkpoint_manager
        self.continue_on_error = continue_on_error

    def process_batches(
        self,
        items: List[T],
        process_batch_fn: Callable[[List[T]], List[T]],
        description: str = "Processing",
        auto_save_frequency: int = 1,
    ) -> List[T]:
        """
        Process items in batches with progress tracking and checkpointing.

        Args:
            items: Items to process
            process_batch_fn: Function that processes a batch and returns results
            description: Description for progress bar
            auto_save_frequency: Save checkpoint every N batches

        Returns:
            List of processed items
        """
        from .progress import get_progress

        # Load existing checkpoint
        all_results = []
        if self.checkpoint_manager:
            all_results = self.checkpoint_manager.load_checkpoint()
            items = self.checkpoint_manager.filter_unprocessed(items)

        if not items:
            logger.info("All items already processed!")
            return all_results

        logger.info(f"Processing {len(items):,} items in batches of {self.batch_size}")
        if (
            self.checkpoint_manager
            and self.checkpoint_manager.was_loaded_from_checkpoint()
        ):
            logger.info(f"Resuming from {len(all_results):,} already processed items")

        # Process in batches with item-level progress tracking
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size

        try:
            with get_progress(description, total=len(items)) as progress:
                task_id = progress.add_task(description, total=len(items))

                for batch_idx in range(total_batches):
                    batch_start = batch_idx * self.batch_size
                    batch_end = min(batch_start + self.batch_size, len(items))
                    batch_items = items[batch_start:batch_end]

                    try:
                        # Process batch
                        batch_results = process_batch_fn(batch_items)

                        # Filter out any results that were already processed
                        # (important for resume scenarios where partial batches might be re-processed)
                        if self.checkpoint_manager:
                            new_results = []
                            for result in batch_results:
                                if not self.checkpoint_manager.is_processed(result):
                                    new_results.append(result)
                                    self.checkpoint_manager.mark_processed(result)
                            batch_results = new_results

                        all_results.extend(batch_results)

                        # Update progress with number of items processed in this batch
                        progress.advance(task_id, advance=len(batch_items))

                        # Auto-save checkpoint
                        if (
                            self.checkpoint_manager
                            and auto_save_frequency > 0
                            and (batch_idx + 1) % auto_save_frequency == 0
                            and batch_results
                        ):
                            # Append only the new results to checkpoint
                            self.checkpoint_manager.append_to_checkpoint(batch_results)

                    except Exception as e:
                        error_msg = f"Error processing batch {batch_idx + 1}/{total_batches}: {e}"
                        logger.error(error_msg)

                        # Still update progress for items we attempted to process
                        progress.advance(task_id, advance=len(batch_items))

                        if self.checkpoint_manager and batch_results:
                            self.checkpoint_manager.append_to_checkpoint(batch_results)

                        if not self.continue_on_error:
                            raise
                        else:
                            logger.warning(
                                "Continuing with next batch due to continue_on_error=True"
                            )
                            continue

        except KeyboardInterrupt:
            logger.info(f"Interrupted after processing {len(all_results):,} items")
            # No need to save - we've been appending as we go
            raise

        logger.info(f"Successfully processed {len(all_results):,} items")
        return all_results


# Convenience functions for common use cases


def create_jsonl_checkpoint_manager(
    checkpoint_path: Optional[str],
    uid_key: str = "uid",
) -> CheckpointManager[Dict[str, Any]]:
    """Create a checkpoint manager for JSONL data (dicts with uid field)."""
    return CheckpointManager(
        checkpoint_path=checkpoint_path,
        get_uid_fn=lambda x: x.get(uid_key, str(hash(str(x)))),
        serialize_fn=lambda x: x,
        deserialize_fn=lambda x: x,
    )


def cleanup_checkpoint_file(checkpoint_path: str, auto_cleanup: bool = False) -> None:
    """
    Clean up checkpoint file with optional auto-cleanup.

    Args:
        checkpoint_path: Path to checkpoint file
        auto_cleanup: If True, automatically delete the file. If False, just show message.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return

    if auto_cleanup:
        try:
            os.remove(checkpoint_path)
            logger.info(f"Cleaned up checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not delete checkpoint {checkpoint_path}: {e}")
    else:
        logger.info(f"You can now delete checkpoint: {checkpoint_path}")


def auto_enable_checkpointing(
    output_path: str,
    checkpoint_path: Optional[str] = None,
    min_items_for_auto_checkpoint: int = 100,
) -> Optional[str]:
    """
    Auto-enable checkpointing for operations above a certain size.

    Args:
        output_path: Main output file path
        checkpoint_path: Explicit checkpoint path (if provided, returned as-is)
        min_items_for_auto_checkpoint: Minimum number of items to auto-enable checkpointing

    Returns:
        Checkpoint path to use (None if checkpointing should be disabled)
    """
    if checkpoint_path is not None:
        return checkpoint_path

    # For now, always auto-enable for operations that specify an output path
    # In the future, this could be made smarter based on estimated processing time
    output_stem = Path(output_path).stem
    return f"{output_stem}_checkpoint.jsonl"
