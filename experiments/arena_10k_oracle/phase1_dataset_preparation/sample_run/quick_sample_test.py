#!/usr/bin/env python3
"""
Quick test to verify the sample configuration works correctly.
Run this before the full sample pipeline to catch any issues early.
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import after path setup
from sample_config import (
    SAMPLE_MODE,
    SAMPLE_SIZE,
    DATA_DIR,
    ARENA_QUESTIONS,
    get_sample_indices,
    log_sample_info,
)

console = Console()


def test_sample_setup():
    """Test that sample configuration is working."""
    console.print(Panel.fit("Testing 1% Sample Configuration", title="🧪 Quick Test"))

    # Show configuration
    log_sample_info()

    # Check directories
    console.print(f"\n📁 Checking directories...")
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {DATA_DIR}")
    else:
        console.print(f"  Exists: {DATA_DIR}")

    # Test sample selection
    console.print(f"\n🎲 Testing sample selection...")
    if SAMPLE_MODE:
        # Create dummy data to test sampling
        dummy_size = 1000
        indices = get_sample_indices(dummy_size)
        console.print(f"  Would select {len(indices)} items from {dummy_size}")
        console.print(f"  First 10 indices: {indices[:10]}")

        # Verify deterministic
        indices2 = get_sample_indices(dummy_size)
        if indices == indices2:
            console.print("  ✅ Sample selection is deterministic")
        else:
            console.print("  ❌ Sample selection is not deterministic!")

    # Check for existing base data
    base_data = Path(__file__).parent.parent / "data" / "arena_questions_base.jsonl"
    if base_data.exists():
        with open(base_data) as f:
            total_lines = sum(1 for _ in f)
        console.print(f"\n📊 Found base dataset: {total_lines} prompts")

        if SAMPLE_MODE:
            sample_indices = get_sample_indices(total_lines)
            console.print(f"  Will sample {len(sample_indices)} prompts")
    else:
        console.print("\n⚠️  Base dataset not found - run 01_prepare_data.py first")

    console.print("\n✅ Configuration test complete!")
    return True


if __name__ == "__main__":
    # Test with sample mode
    import os

    os.environ["ARENA_SAMPLE_MODE"] = "true"
    os.environ["ARENA_SAMPLE_SIZE"] = "100"

    test_sample_setup()
