"""Utilities for working with fresh draws in DR estimation."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import logging

from ..data.fresh_draws import FreshDrawSample, FreshDrawDataset
from ..data.models import Dataset
from ..data.loaders import FreshDrawLoader

logger = logging.getLogger(__name__)


def load_fresh_draws_from_jsonl(path: str) -> Dict[str, FreshDrawDataset]:
    """Load fresh draws from JSONL file, grouped by policy.

    This function delegates to FreshDrawLoader in the data module
    for consistency with other data loading operations.

    Expected JSONL format:
    {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.85, "draw_idx": 0}
    {"prompt_id": "0", "target_policy": "premium", "judge_score": 0.82, "draw_idx": 1}
    {"prompt_id": "1", "target_policy": "premium", "judge_score": 0.90, "draw_idx": 0}

    Args:
        path: Path to JSONL file containing fresh draws

    Returns:
        Dict mapping policy names to FreshDrawDataset objects
    """
    return FreshDrawLoader.load_from_jsonl(path)


def validate_fresh_draws(
    fresh_draws: FreshDrawDataset,
    logged_dataset: Dataset,
    policy: str,
) -> None:
    """Validate fresh draws have complete coverage for a policy.

    Args:
        fresh_draws: Fresh draw dataset to validate
        logged_dataset: Logged dataset to check coverage against
        policy: Target policy name

    Raises:
        ValueError: If fresh draws don't have complete coverage
    """
    # Get valid samples for this policy from logged data
    valid_samples = [
        s for s in logged_dataset.samples if s.get_importance_weight(policy) is not None
    ]

    if not valid_samples:
        raise ValueError(f"No valid logged samples for policy '{policy}'")

    # Get prompt IDs
    logged_ids = {s.prompt_id for s in valid_samples}
    fresh_ids = set(fresh_draws.get_prompt_ids())

    # Check coverage
    missing = logged_ids - fresh_ids
    extra = fresh_ids - logged_ids

    if missing:
        raise ValueError(
            f"Fresh draws missing for {len(missing)} prompts:\n"
            f"  First 5 missing: {list(missing)[:5]}\n"
            f"DR requires fresh draws for ALL samples with valid importance weights."
        )

    if extra:
        logger.warning(
            f"Fresh draws contain {len(extra)} extra prompts not in logged data. "
            f"These will be ignored."
        )

    # Check draws per prompt consistency
    for prompt_id in logged_ids:
        try:
            prompt_id_str = str(prompt_id) if prompt_id is not None else ""
            scores = fresh_draws.get_scores_for_prompt_id(prompt_id_str)
            if len(scores) != fresh_draws.draws_per_prompt:
                raise ValueError(
                    f"Prompt '{prompt_id}' has {len(scores)} draws, "
                    f"expected {fresh_draws.draws_per_prompt}"
                )
        except ValueError as e:
            raise ValueError(f"Validation failed: {e}")

    logger.info(
        f"Fresh draws validated: {len(fresh_ids)} prompts, "
        f"{fresh_draws.draws_per_prompt} draws/prompt"
    )


def create_synthetic_fresh_draws(
    logged_dataset: Dataset,
    target_policy: str,
    draws_per_prompt: int = 5,
    score_correlation: float = 0.8,
    seed: Optional[int] = None,
) -> FreshDrawDataset:
    """Create synthetic fresh draws for testing.

    Generates correlated judge scores based on logged data,
    useful for testing DR without actual API calls.

    Args:
        logged_dataset: Logged dataset to base fresh draws on
        target_policy: Target policy name
        draws_per_prompt: Number of draws per prompt
        score_correlation: Correlation with logged judge scores (0-1)
        seed: Random seed for reproducibility

    Returns:
        Synthetic FreshDrawDataset
    """
    if seed is not None:
        np.random.seed(seed)

    # Get valid samples for this policy
    valid_samples = [
        s
        for s in logged_dataset.samples
        if s.get_importance_weight(target_policy) is not None
    ]

    if not valid_samples:
        raise ValueError(f"No valid samples for policy '{target_policy}'")

    samples: List[FreshDrawSample] = []
    for sample in valid_samples:
        prompt_id = sample.prompt_id
        base_score = sample.metadata.get("judge_score", 0.5)

        for draw_idx in range(draws_per_prompt):
            # Generate correlated score
            noise = np.random.normal(0, 0.1 * (1 - score_correlation))
            score = np.clip(base_score + noise, 0, 1)

            fresh_sample = FreshDrawSample(
                prompt_id=prompt_id,
                target_policy=target_policy,
                judge_score=float(score),
                response=f"Synthetic response for {prompt_id} draw {draw_idx}",
                draw_idx=draw_idx,
                fold_id=None,
            )
            samples.append(fresh_sample)

    dataset = FreshDrawDataset(
        target_policy=target_policy,
        draws_per_prompt=draws_per_prompt,
        samples=samples,
    )

    logger.info(
        f"Created synthetic fresh draws: {len(samples)} samples, "
        f"{len(valid_samples)} prompts, {draws_per_prompt} draws/prompt"
    )

    return dataset


def load_fresh_draws_auto(
    data_dir: Path,
    policy: str,
    verbose: bool = False,
) -> FreshDrawDataset:
    """
    Load fresh draws from files.

    This function tries to load fresh draws from standard locations:
    1. {data_dir}/{policy}_responses.jsonl
    2. {data_dir}/responses/{policy}_responses.jsonl
    3. {data_dir}/{policy}_fresh.jsonl
    4. {data_dir}/fresh_draws/{policy}.jsonl

    Args:
        data_dir: Directory to search for fresh draw files
        policy: Target policy name
        verbose: Whether to log detailed information

    Returns:
        FreshDrawDataset for the specified policy

    Raises:
        FileNotFoundError: If no fresh draw file found
    """
    # Standard file patterns to check
    possible_files = [
        data_dir / f"{policy}_responses.jsonl",
        data_dir / "responses" / f"{policy}_responses.jsonl",
        data_dir / f"{policy}_fresh.jsonl",
        data_dir / "fresh_draws" / f"{policy}.jsonl",
    ]

    # Try to load from each possible location
    for file_path in possible_files:
        if file_path.exists():
            if verbose:
                logger.info(f"Loading fresh draws from {file_path}")

            try:
                # Load the file
                fresh_samples = []
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)

                        # Handle different formats
                        fresh_sample = FreshDrawSample(
                            prompt_id=str(data.get("prompt_id")),
                            target_policy=policy,
                            response=data.get("response", ""),
                            judge_score=data.get("judge_score")
                            or data.get("metadata", {}).get("judge_score", 0.5),
                            draw_idx=data.get("draw_idx", 0),
                            fold_id=data.get("fold_id"),
                        )
                        fresh_samples.append(fresh_sample)

                # Create dataset
                fresh_dataset = FreshDrawDataset(
                    target_policy=policy,
                    draws_per_prompt=1,  # Will be updated based on actual data
                    samples=fresh_samples,
                )

                # Update draws_per_prompt based on actual data
                prompt_counts: Dict[str, int] = {}
                for sample in fresh_samples:
                    prompt_counts[sample.prompt_id] = (
                        prompt_counts.get(sample.prompt_id, 0) + 1
                    )
                if prompt_counts:
                    fresh_dataset.draws_per_prompt = max(prompt_counts.values())

                if verbose:
                    logger.info(
                        f"Loaded {len(fresh_samples)} fresh draws for {policy} "
                        f"({len(prompt_counts)} unique prompts)"
                    )

                return fresh_dataset

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

    # No file found - raise error with helpful message
    searched_paths = "\n  ".join(str(p) for p in possible_files)
    raise FileNotFoundError(
        f"No fresh draw file found for policy '{policy}'. Searched:\n  {searched_paths}\n"
        f"Fresh draws must be generated from real teacher forcing responses."
    )


def save_fresh_draws_to_jsonl(
    datasets: Dict[str, FreshDrawDataset],
    path: str,
) -> None:
    """Save fresh draw datasets to JSONL file.

    Args:
        datasets: Dict mapping policy names to FreshDrawDataset objects
        path: Output path for JSONL file
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w") as f:
        for policy, dataset in datasets.items():
            for sample in dataset.samples:
                record = {
                    "prompt_id": sample.prompt_id,
                    "target_policy": sample.target_policy,
                    "judge_score": sample.judge_score,
                    "draw_idx": sample.draw_idx,
                }
                if sample.response is not None:
                    record["response"] = sample.response

                f.write(json.dumps(record) + "\n")

    total_samples = sum(len(d.samples) for d in datasets.values())
    logger.info(f"Saved {total_samples} fresh draws to {path_obj}")
