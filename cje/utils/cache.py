"""Simple cache utilities for CJE.

Just the essentials: safe file naming and basic pickle operations.
"""

from pathlib import Path
from typing import Any, Optional
import pickle
import logging

logger = logging.getLogger(__name__)


def sanitize_model_name(name: str) -> str:
    """Convert model names to filesystem-safe cache keys.

    Args:
        name: Raw name (e.g., "accounts/fireworks/models/llama-v3")

    Returns:
        Safe name (e.g., "accounts_fireworks_models_llama_v3")
    """
    return (
        name.replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
        .lower()
    )


def get_cache_path(cache_dir: Path, prefix: str, model_name: str) -> Path:
    """Get cache file path with safe naming.

    Args:
        cache_dir: Cache directory
        prefix: File prefix (e.g., "oracle_labels", "proxy_scores")
        model_name: Model name to sanitize

    Returns:
        Path to cache file
    """
    safe_name = sanitize_model_name(model_name)
    return cache_dir / f"{prefix}_{safe_name}.pkl"


def save_cache(data: Any, cache_dir: Path, prefix: str, model_name: str) -> Path:
    """Save data to cache file.

    Args:
        data: Data to cache
        cache_dir: Cache directory
        prefix: File prefix
        model_name: Model name

    Returns:
        Path where data was saved
    """
    cache_path = get_cache_path(cache_dir, prefix, model_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)

    logger.debug(f"Saved cache: {cache_path}")
    return cache_path


def load_cache(cache_dir: Path, prefix: str, model_name: str) -> Optional[Any]:
    """Load data from cache file.

    Args:
        cache_dir: Cache directory
        prefix: File prefix
        model_name: Model name

    Returns:
        Cached data if found, None otherwise
    """
    cache_path = get_cache_path(cache_dir, prefix, model_name)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"Loaded cache: {cache_path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None


def cache_exists(cache_dir: Path, prefix: str, model_name: str) -> bool:
    """Check if cache file exists.

    Args:
        cache_dir: Cache directory
        prefix: File prefix
        model_name: Model name

    Returns:
        True if cache exists
    """
    cache_path = get_cache_path(cache_dir, prefix, model_name)
    return cache_path.exists()
