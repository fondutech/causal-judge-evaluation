#!/usr/bin/env python3
"""
Convert arena_prompts_10k.jsonl to arena_questions_base.jsonl format.
"""

import json
from pathlib import Path


def convert_format():
    """Convert the existing arena data to the expected format."""
    input_file = Path(__file__).parent.parent / "data" / "arena_prompts_10k.jsonl"
    output_file = Path(__file__).parent.parent / "data" / "arena_questions_base.jsonl"

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return

    print(f"Converting {input_file} to {output_file}")

    with open(input_file) as f_in, open(output_file, "w") as f_out:
        for i, line in enumerate(f_in):
            data = json.loads(line)
            # The format is already correct, just ensure we have the right fields
            output = {
                "prompt_id": data.get("prompt_id", f"arena_{i}"),
                "prompt": data["prompt"],
                "metadata": data.get("metadata", {}),
            }
            f_out.write(json.dumps(output) + "\n")

    # Count lines
    with open(output_file) as f:
        count = sum(1 for _ in f)

    print(f"✅ Converted {count} prompts")
    print(f"📁 Output saved to: {output_file}")


if __name__ == "__main__":
    convert_format()
