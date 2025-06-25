#!/usr/bin/env python3
"""
Fix duplicate entries in scored data files.
Removes duplicates based on prompt_id, keeping the first occurrence.
"""

import json
import sys
from pathlib import Path
from collections import OrderedDict


def fix_duplicates(input_file: str, output_file: str = None) -> int:
    """Remove duplicate entries from JSONL file based on prompt_id."""
    if output_file is None:
        output_file = input_file

    # Read all entries
    seen_ids = OrderedDict()
    duplicates = 0

    print(f"Reading {input_file}...")
    with open(input_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                prompt_id = entry.get("prompt_id")

                if prompt_id is None:
                    print(f"Warning: Line {line_num} has no prompt_id")
                    continue

                if prompt_id not in seen_ids:
                    seen_ids[prompt_id] = entry
                else:
                    duplicates += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    # Write deduplicated entries
    print(f"Writing {len(seen_ids)} unique entries to {output_file}...")
    with open(output_file, "w") as f:
        for entry in seen_ids.values():
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Removed {duplicates} duplicates")
    print(f"📊 Final count: {len(seen_ids)} unique entries")

    return len(seen_ids)


def main() -> None:
    """Fix duplicates in all scored files."""
    data_dir = Path("data")

    files_to_fix = [
        "p0_scored_deterministic.jsonl",
        "p0_scored_uncertainty.jsonl",
        "p0_scored_deterministic.checkpoint.jsonl",
        "p0_scored_uncertainty.checkpoint.jsonl",
        "targets_scored_deterministic.jsonl",
        "targets_scored_uncertainty.jsonl",
        "targets_scored_deterministic.checkpoint.jsonl",
        "targets_scored_uncertainty.checkpoint.jsonl",
    ]

    for filename in files_to_fix:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\n🔧 Fixing {filename}...")
            fix_duplicates(str(filepath))
        else:
            print(f"\n⏭️  Skipping {filename} (not found)")


if __name__ == "__main__":
    main()
