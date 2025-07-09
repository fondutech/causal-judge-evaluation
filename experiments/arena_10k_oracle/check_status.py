#!/usr/bin/env python3
"""Quick status check for Arena 10K pipeline."""

import json
from pathlib import Path
from rich.console import Console

console = Console()

# Check file counts
files = {
    "Prompts": "phase1_dataset_preparation/data/arena_prompts_10k.jsonl",
    "Responses": "phase1_dataset_preparation/data/all_responses.jsonl",
    "Log probs": "phase1_dataset_preparation/data/logprobs.jsonl",
    "P0 scored (det)": "data/p0_scored_deterministic.jsonl",
    "Targets scored (det)": "data/targets_scored_deterministic.jsonl",
}

console.print("[bold]Arena 10K Pipeline Status[/bold]\n")

for name, path in files.items():
    if Path(path).exists():
        count = sum(1 for _ in open(path))
        console.print(f"✅ {name}: {count:,} items")
    else:
        console.print(f"❌ {name}: not found")

# Check for checkpoint
if Path("phase1_dataset_preparation/.pipeline_checkpoint.pkl").exists():
    console.print("\n📌 Pipeline checkpoint found (can resume)")

# Check for extreme weights log
if Path("phase1_dataset_preparation/data/extreme_weights.jsonl").exists():
    extreme_count = sum(
        1 for _ in open("phase1_dataset_preparation/data/extreme_weights.jsonl")
    )
    console.print(f"\n⚠️  Extreme weights detected: {extreme_count} issues logged")
