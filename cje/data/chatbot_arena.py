"""LMSYS Chatbot Arena dataset adapter for CJE."""

from pathlib import Path
from typing import Optional, Iterator, Dict
import json

import pyarrow as pa
from tqdm import tqdm

from .base import CJEDataset
from .schema import CJESample


class ChatbotArenaDataset(CJEDataset):
    """Adapter for LMSYS Chatbot Arena conversations dataset.

    This dataset extracts unique contexts from arena conversations for fresh response generation.
    It does NOT use existing arena responses, ensuring proper policy comparison rather than
    importance sampling on pre-existing data.
    """

    name = "ChatbotArena"

    def __init__(self, tbl: pa.Table) -> None:
        """Initialize the dataset.

        Args:
            tbl: PyArrow table containing the dataset
        """
        self._tbl = tbl

    @classmethod
    def download(
        cls,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        dataset_name: str = "agie-ai/lmsys-chatbot_arena_conversations",
    ) -> "ChatbotArenaDataset":
        """Download and process the Chatbot Arena dataset.

        This always operates in context-only mode, extracting contexts for fresh response generation.

        Args:
            cache_dir: Directory to cache the dataset. If None, uses default cache.
            split: Dataset split to load. If None, loads all available data.
            dataset_name: Name of the HuggingFace dataset to load

        Returns:
            A ChatbotArenaDataset instance.
        """
        from datasets import load_dataset
        import pyarrow.parquet as pq

        cache_dir_path = (
            Path(cache_dir) if cache_dir else Path.home() / ".cache" / "cje"
        )
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        if split:
            parquet_path = cache_dir_path / f"chatbot_arena_{split}.parquet"
        else:
            parquet_path = cache_dir_path / "chatbot_arena_full.parquet"

        if not parquet_path.exists():
            if split is None:
                ds = load_dataset(dataset_name)
                from datasets import concatenate_datasets

                all_datasets = []
                for split_name, dataset in ds.items():
                    dataset_with_split = dataset.add_column(
                        "split", [split_name] * len(dataset)
                    )
                    all_datasets.append(dataset_with_split)
                full_dataset = concatenate_datasets(all_datasets)
                full_dataset.to_parquet(str(parquet_path))
            else:
                ds = load_dataset(dataset_name, split=split)
                ds_with_split = ds.add_column("split", [split] * len(ds))
                ds_with_split.to_parquet(str(parquet_path))

        tbl = pq.read_table(str(parquet_path))
        return cls(tbl)

    def itersamples(self, disable_progress: bool = False) -> Iterator[CJESample]:
        """Yield CJESamples for policy comparison.

        Extracts unique contexts and yields them with empty responses,
        forcing fresh response generation from the specified policies. This enables proper
        policy comparison rather than importance sampling on existing arena responses.

        Args:
            disable_progress: If True, disable the progress bar.
        """
        seen_contexts = set()  # Track unique contexts to avoid duplicates

        for row in tqdm(self._tbl.to_pylist(), disable=disable_progress):
            conv_id = row.get("conversation_id", row.get("question_id", None))
            if conv_id is None:
                continue

            # Extract the conversation turns
            conversation_a = row.get("conversation_a", [])

            # Build context from the human parts of the conversation
            context_parts = []
            for turn in conversation_a:
                if turn.get("role") == "user":
                    context_parts.append(turn.get("content", ""))
            context = "\n".join(context_parts)

            # Only yield unique contexts for fresh response generation
            # This is the correct approach for arena analysis where we compare fresh responses
            # from our specified policies rather than doing importance sampling on existing responses
            if context in seen_contexts:
                continue  # Skip duplicate contexts
            seen_contexts.add(context)

            yield CJESample(
                uid=f"{conv_id}_arena_context",
                context=context,
                response="",  # Empty response triggers fresh generation in pipeline
                y_true=None,  # Will be filled by oracle if enabled
                logp=None,
                meta={
                    "conversation_id": conv_id,
                    "language": row.get("language", "unknown"),
                    "turn": row.get("turn", len(conversation_a) // 2),
                    "split": row.get("split", "unknown"),
                    "anon": row.get("anon", True),
                    "tstamp": row.get("tstamp", None),
                    "arena_policy_comparison": True,
                },
            )
