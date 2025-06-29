"""
Configuration for 1% sample run.

This configuration file allows running the Phase 1 pipeline with a small sample
without modifying the original scripts.
"""

import os
from pathlib import Path

# Sample run configuration
SAMPLE_MODE = os.getenv("ARENA_SAMPLE_MODE", "false").lower() == "true"
SAMPLE_SIZE = int(os.getenv("ARENA_SAMPLE_SIZE", "100"))
SAMPLE_SEED = int(os.getenv("ARENA_SAMPLE_SEED", "42"))

# Directory configuration
if SAMPLE_MODE:
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "sample_1pct"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Adjusted batch sizes for sample run
    BATCH_SIZES = {
        "p0_generation": 10,
        "target_generation": 10,
        "teacher_forcing": 5,
        "oracle_labeling": 20,
        "judge_scoring": 16,
    }

    # File name suffix
    FILE_SUFFIX = "_sample"
else:
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    BATCH_SIZES = {
        "p0_generation": 50,
        "target_generation": 50,
        "teacher_forcing": 10,
        "oracle_labeling": 100,
        "judge_scoring": 50,
    }
    FILE_SUFFIX = ""

# Common file paths
ARENA_QUESTIONS = DATA_DIR / f"arena_questions_base{FILE_SUFFIX}.jsonl"
P0_REPLIES = DATA_DIR / f"p0_replies{FILE_SUFFIX}.jsonl"
TARGET_RESPONSES = DATA_DIR / f"target_responses{FILE_SUFFIX}.jsonl"
P0_WITH_LOGPS = DATA_DIR / f"p0_with_target_logps{FILE_SUFFIX}.jsonl"
ORACLE_LABELS = DATA_DIR / f"oracle_labels{FILE_SUFFIX}.jsonl"
JUDGE_SCORES = DATA_DIR / f"judge_scores{FILE_SUFFIX}.jsonl"

# API configuration
API_CONFIG = {
    "retry_attempts": 3,
    "timeout": 30,
    "rate_limit_delay": 1.0 if SAMPLE_MODE else 0.1,
}

# Cost tracking
COST_PER_1K_TOKENS = {
    "llama4-scout-instruct-basic": 0.20,
    "llama4-maverick-instruct-basic": 0.40,
    "gpt-4o": 2.50,
}


def get_sample_indices(total_size: int) -> list[int]:
    """Get indices for sample selection."""
    if not SAMPLE_MODE:
        return list(range(total_size))

    import random

    random.seed(SAMPLE_SEED)
    return sorted(random.sample(range(total_size), min(SAMPLE_SIZE, total_size)))


def log_sample_info():
    """Log information about sample mode."""
    if SAMPLE_MODE:
        print(f"🔬 SAMPLE MODE: Processing {SAMPLE_SIZE} samples")
        print(f"📁 Output directory: {DATA_DIR}")
        print(f"🎲 Random seed: {SAMPLE_SEED}")
    else:
        print("🚀 FULL MODE: Processing entire dataset")
        print(f"📁 Output directory: {DATA_DIR}")
