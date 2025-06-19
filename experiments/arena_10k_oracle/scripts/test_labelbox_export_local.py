#!/usr/bin/env python3
"""Test Labelbox export format locally without API calls."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import json
import pandas as pd
from cje.utils.progress import console

# Test the export format creation
console.print("🔬 Testing Labelbox export format creation...")

# Create sample data
sample_p0 = pd.DataFrame(
    {
        "prompt_id": ["p001", "p002", "p003"],
        "prompt": ["What is 2+2?", "Explain quantum physics", "How do I make a cake?"],
        "response": [
            "2+2 equals 4.",
            "Quantum physics is the study of matter and energy at the smallest scales.",
            "To make a cake, you'll need flour, eggs, sugar, and butter. Mix ingredients and bake at 350°F.",
        ],
    }
)

sample_target = pd.DataFrame(
    {
        "prompt_id": ["p001", "p002"],
        "prompt": ["What is 2+2?", "Explain quantum physics"],
        "response": [
            "The sum of 2 and 2 is 4.",
            "Quantum physics deals with phenomena at atomic and subatomic scales where classical physics breaks down.",
        ],
        "policy": ["target", "target"],
    }
)

# Test the data preparation logic
import random

random.seed(42)

# Mark π₀ data
sample_p0["policy"] = "pi_0"
sample_p0["split"] = "evaluation"

# Sample calibration (1 out of 3)
calibration_idx = 0  # First sample for calibration
sample_p0.loc[calibration_idx, "split"] = "calibration"

# Get calibration samples
calibration_df = sample_p0[sample_p0["split"] == "calibration"].copy()

# Combine
combined_df = pd.concat([calibration_df, sample_target], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

console.print(f"\n📊 Combined data ({len(combined_df)} rows):")
for idx, row in combined_df.iterrows():
    console.print(f"  {idx}: {row['policy']} - {row['prompt'][:30]}...")

# Create Labelbox format
assets = []
tracking_data = []

for idx, row in combined_df.iterrows():
    task_id = f"arena_task_{idx:06d}"

    # Format the conversation
    conversation_text = f"USER: {row['prompt']}\n\nASSISTANT: {row['response']}"

    # Create Labelbox data row
    asset = {
        "row_data": conversation_text,
        "global_key": task_id,
        "media_type": "TEXT",
        "metadata_fields": [
            {"name": "task_type", "value": "conversation_rating"},
            {"name": "prompt_length", "value": len(row["prompt"])},
            {"name": "response_length", "value": len(row["response"])},
        ],
        "attachments": [
            {"type": "RAW_TEXT", "value": f"Prompt: {row['prompt']}"},
            {"type": "RAW_TEXT", "value": f"Response: {row['response']}"},
        ],
    }
    assets.append(asset)

    # Internal tracking
    tracking_item = {
        "task_id": task_id,
        "prompt_id": row.get("prompt_id", ""),
        "policy": row.get("policy", "unknown"),
        "split": row.get("split", "evaluation"),
    }
    tracking_data.append(tracking_item)

console.print(f"\n✅ Created {len(assets)} Labelbox assets")

# Show sample asset
console.print("\n📝 Sample Labelbox asset:")
sample_asset = assets[0]
console.print(f"  Global key: {sample_asset['global_key']}")
console.print(f"  Text preview: {sample_asset['row_data'][:100]}...")
console.print(f"  Metadata: {sample_asset['metadata_fields']}")

# Show tracking
tracking_df = pd.DataFrame(tracking_data)
console.print("\n🔍 Tracking data:")
for policy, count in tracking_df["policy"].value_counts().items():
    console.print(f"  {policy}: {count}")

# Save sample files
output_dir = Path("../data/labeling/test")
output_dir.mkdir(parents=True, exist_ok=True)

# Save sample assets
sample_file = output_dir / "sample_labelbox_assets.json"
with open(sample_file, "w") as f:
    json.dump(assets[:2], f, indent=2)
console.print(f"\n💾 Saved sample assets to: {sample_file}")

# Save tracking
tracking_file = output_dir / "sample_tracking.jsonl"
with open(tracking_file, "w") as f:
    for item in tracking_data:
        f.write(json.dumps(item) + "\n")
console.print(f"💾 Saved tracking to: {tracking_file}")

console.print(
    "\n✅ Test complete! The export format is ready for use with Labelbox API."
)
