"""
Modular component caching system for CJE pipeline.

This module provides efficient caching of pipeline stages using content-based hashing
to enable true incremental reuse of computation. Each stage is cached independently,
allowing changes to one component without invalidating others.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


def _convert_to_serializable(obj: Any) -> Any:
    """Convert any object to a JSON-serializable format."""
    try:
        # Try importing OmegaConf to handle DictConfig objects
        from omegaconf import DictConfig, ListConfig

        if isinstance(obj, (DictConfig, ListConfig)):
            from omegaconf import OmegaConf

            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass

    if hasattr(obj, "__dict__"):
        # For objects with attributes, extract key fields
        return {
            k: _convert_to_serializable(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    elif isinstance(obj, dict):
        return {str(k): _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For other types, try to convert to string
        return str(obj)


def compute_content_hash(data: Any) -> str:
    """Compute a stable hash of any data structure."""
    # Convert to JSON-serializable format first
    serializable_data = _convert_to_serializable(data)
    # Convert to JSON with sorted keys for stable hashing
    json_str = json.dumps(serializable_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:12]


def get_cache_dir(work_dir: Path) -> Path:
    """Get the cache directory, creating it if needed."""
    cache_dir = work_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_chunk_path(work_dir: Path, stage: str, content_hash: str) -> Path:
    """Get the path for a cached chunk file."""
    cache_dir = get_cache_dir(work_dir)
    stage_dir = cache_dir / stage
    stage_dir.mkdir(exist_ok=True)
    return stage_dir / f"{content_hash}.jsonl"


def chunk_exists(work_dir: Path, stage: str, content_hash: str) -> bool:
    """Check if a cached chunk exists."""
    chunk_path = get_chunk_path(work_dir, stage, content_hash)
    return chunk_path.exists()


def save_chunk(
    work_dir: Path,
    stage: str,
    content_hash: str,
    rows: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save rows to a cached chunk file."""
    chunk_path = get_chunk_path(work_dir, stage, content_hash)

    # Create metadata file alongside the chunk
    meta_path = chunk_path.with_suffix(".meta.json")
    chunk_metadata = {
        "stage": stage,
        "content_hash": content_hash,
        "row_count": len(rows),
        "created_at": str(Path().stat().st_mtime) if chunk_path.exists() else None,
        **(metadata or {}),
    }

    # Convert metadata to serializable format
    serializable_metadata = _convert_to_serializable(chunk_metadata)

    with open(meta_path, "w") as f:
        json.dump(serializable_metadata, f, indent=2)

    # Save the actual data
    with open(chunk_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    logger.info(f"Saved {len(rows)} rows to cache: {stage}/{content_hash}")
    return chunk_path


def load_chunk(work_dir: Path, stage: str, content_hash: str) -> List[Dict[str, Any]]:
    """Load rows from a cached chunk file."""
    chunk_path = get_chunk_path(work_dir, stage, content_hash)

    if not chunk_path.exists():
        raise FileNotFoundError(f"Cache chunk not found: {chunk_path}")

    rows = []
    with open(chunk_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    logger.info(f"Loaded {len(rows)} rows from cache: {stage}/{content_hash}")
    return rows


def get_chunk_metadata(
    work_dir: Path, stage: str, content_hash: str
) -> Optional[Dict[str, Any]]:
    """Load metadata for a cached chunk."""
    chunk_path = get_chunk_path(work_dir, stage, content_hash)
    meta_path = chunk_path.with_suffix(".meta.json")

    if not meta_path.exists():
        return None

    with open(meta_path, "r") as f:
        data: Dict[str, Any] = json.load(f)  # Explicit type annotation
        return data  # Explicitly return the loaded data


class CacheConfig:
    """Configuration object for computing cache keys."""

    def __init__(self, **kwargs: Any) -> None:
        # Store all config values in a normalized way
        self.data = self._normalize_config(kwargs)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config for stable hashing."""
        result: Dict[str, Any] = _convert_to_serializable(
            config
        )  # Explicit type annotation
        return result  # Explicitly return the serialized config

    def hash(self) -> str:
        """Compute hash of this configuration."""
        return compute_content_hash(self.data)

    def __repr__(self) -> str:
        return f"CacheConfig(hash={self.hash()}, keys={list(self.data.keys())})"


# Stage-specific hash computation functions
def compute_contexts_hash(
    dataset_config: Dict[str, Any], logging_policy_config: Dict[str, Any]
) -> str:
    """Compute hash for the base contexts stage."""
    config = CacheConfig(dataset=dataset_config, logging_policy=logging_policy_config)
    return config.hash()


def compute_judge_hash(contexts_hash: str, judge_config: Dict[str, Any]) -> str:
    """Compute hash for the judge scores stage."""
    config = CacheConfig(contexts_hash=contexts_hash, judge=judge_config)
    return config.hash()


def compute_calibration_hash(judge_hash: str, oracle_config: Dict[str, Any]) -> str:
    """Compute hash for the calibration stage."""
    config = CacheConfig(judge_hash=judge_hash, oracle=oracle_config)
    return config.hash()


def compute_target_logprobs_hash(
    contexts_hash: str, target_policies_config: List[Dict[str, Any]]
) -> str:
    """Compute hash for the target log probabilities stage."""
    config = CacheConfig(
        contexts_hash=contexts_hash, target_policies=target_policies_config
    )
    return config.hash()


def compute_oracle_hash(judge_hash: str, oracle_config: Dict[str, Any]) -> str:
    """Compute hash for the oracle labels stage."""
    config = CacheConfig(judge_hash=judge_hash, oracle=oracle_config)
    return config.hash()


def list_cached_stages(work_dir: Path) -> Dict[str, List[str]]:
    """List all cached stages and their content hashes."""
    cache_dir = get_cache_dir(work_dir)
    stages = {}

    for stage_dir in cache_dir.iterdir():
        if stage_dir.is_dir():
            stage_name = stage_dir.name
            hashes = []
            for chunk_file in stage_dir.glob("*.jsonl"):
                hashes.append(chunk_file.stem)
            stages[stage_name] = sorted(hashes)

    return stages


def clear_cache(work_dir: Path, stage: Optional[str] = None) -> int:
    """Clear cache files. Returns number of files deleted."""
    cache_dir = get_cache_dir(work_dir)

    if stage:
        stage_dir = cache_dir / stage
        if stage_dir.exists():
            files_deleted = 0
            for file_path in stage_dir.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    files_deleted += 1
            stage_dir.rmdir()
            logger.info(
                f"Cleared cache for stage '{stage}': {files_deleted} files deleted"
            )
            return files_deleted
    else:
        files_deleted = 0
        for file_path in cache_dir.rglob("*"):
            if file_path.is_file():
                file_path.unlink()
                files_deleted += 1
        # Remove empty directories
        for dir_path in sorted(cache_dir.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
        logger.info(f"Cleared entire cache: {files_deleted} files deleted")
        return files_deleted

    return 0
