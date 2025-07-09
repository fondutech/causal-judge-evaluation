#!/usr/bin/env python3
"""Test continuation method specifically."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cje.utils.teacher_forcing import RobustTeacherForcing

# Set up API key
if not os.getenv("FIREWORKS_API_KEY"):
    print("Error: FIREWORKS_API_KEY not set")
    sys.exit(1)

test_case = {
    "id": "arena_sampled_51",
    "prompt": "Bob is a young man with a terribly vivid imagination, full of stories he keeps for himself. Indeed, the fiction he invents is not for the faint of heart, since it is always full of illicit and improper sexual fantasies. Write a single dot.",
    "response": "."
}

model = "accounts/fireworks/models/llama4-maverick-instruct-basic"

# Create teacher forcing instance
tf = RobustTeacherForcing(
    provider="fireworks",
    model=model,
    temperature=0.0
)

print(f"Testing: {test_case['id']}")
print(f"Prompt length: {len(test_case['prompt'])} chars")
print(f"Response: '{test_case['response']}' (length: {len(test_case['response'])} chars)")
print("=" * 80)

# Test continuation method directly
result = tf._continuation_method(test_case['prompt'], test_case['response'])

print(f"\nContinuation method result:")
print(f"Status: {result.status}")
print(f"Is valid: {result.is_valid}")
if result.is_valid:
    print(f"Value: {result.value}")
else:
    print(f"Error: {result.error}")
print(f"Metadata: {result.metadata}")

# Now test the full compute_log_prob to see where it fails
print("\n" + "=" * 80)
print("Testing full compute_log_prob:")
full_result = tf.compute_log_prob(test_case['prompt'], test_case['response'])
print(f"Status: {full_result.status}")
print(f"Is valid: {full_result.is_valid}")
if full_result.is_valid:
    print(f"Value: {full_result.value}")
else:
    print(f"Error: {full_result.error}")
print(f"Metadata: {full_result.metadata}")

# Test with a control case
print("\n" + "=" * 80)
print("Testing control case (should work):")
control_result = tf.compute_log_prob("What is 2+2?", "4")
print(f"Status: {control_result.status}")
print(f"Is valid: {control_result.is_valid}")
if control_result.is_valid:
    print(f"Value: {control_result.value}")
print(f"Metadata: {control_result.metadata}")