#!/usr/bin/env python3
"""Test the updated sampling logic for export scripts."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import json
import pandas as pd
from cje.utils.progress import console

# Load actual data to test sampling
console.print("🔬 Testing sampling logic with actual data...")

# Load π₀ responses
p0_data = []
with open("../data/p0_replies.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i < 100:  # Just load first 100 for testing
            p0_data.append(json.loads(line))
p0_df = pd.DataFrame(p0_data)

# Load target responses
target_data = []
with open("../data/target_ground_truth.jsonl", "r") as f:
    for line in f:
        target_data.append(json.loads(line))
target_df = pd.DataFrame(target_data)

console.print(f"\n📊 Data loaded:")
console.print(f"   • π₀ responses: {len(p0_df):,} (loaded subset)")
console.print(f"   • Target responses: {len(target_df):,}")

# Check target policies
console.print(f"\n🔍 Target policies found:")
for policy, count in target_df["policy"].value_counts().items():
    console.print(f"   • {policy}: {count:,}")

# Test sampling logic
import random

random.seed(42)

# Sample calibration
calibration_fraction = 0.25
n_calibration = int(len(p0_df) * calibration_fraction)
console.print(
    f"\n📐 Sampling {n_calibration} calibration samples from π₀ ({calibration_fraction:.0%})"
)

# Sample from each target policy
target_samples_per_policy = 500
target_policies = [p for p in target_df["policy"].unique() if p != "pi_clone"]

console.print(f"\n📊 Sampling {target_samples_per_policy} from each target policy:")
total_target_samples = 0
for policy in target_policies:
    policy_df = target_df[target_df["policy"] == policy]
    n_samples = min(len(policy_df), target_samples_per_policy)
    total_target_samples += n_samples
    console.print(f"   • {policy}: {n_samples} samples")

# Total samples for labeling
total_samples = n_calibration + total_target_samples
console.print(f"\n📊 Total samples for labeling: {total_samples:,}")
console.print(f"   • Calibration (π₀): {n_calibration:,}")
console.print(f"   • Target policies: {total_target_samples:,}")

# Cost estimate with new numbers
votes_per_sample = 3
cost_per_vote = 0.08
total_votes = total_samples * votes_per_sample
total_cost = total_votes * cost_per_vote

console.print(f"\n💰 Updated cost estimate:")
console.print(f"   • Total samples: {total_samples:,}")
console.print(f"   • Votes per sample: {votes_per_sample}")
console.print(f"   • Total votes: {total_votes:,}")
console.print(f"   • Cost per vote: ${cost_per_vote:.2f}")
console.print(f"   • Total cost: ${total_cost:,.2f}")
console.print(f"   • Time estimate: {total_votes * 45 / 3600:.1f} hours (@ 45s/vote)")

console.print("\n✅ Sampling logic test complete!")
