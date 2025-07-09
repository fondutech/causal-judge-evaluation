#!/usr/bin/env python3
"""Test teacher forcing on the failing prompts."""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cje.utils.teacher_forcing import RobustTeacherForcing, compute_teacher_forced_logprob
from cje.types import LogProbResult

# Set up API key
if not os.getenv("FIREWORKS_API_KEY"):
    print("Error: FIREWORKS_API_KEY not set. Please run:")
    print("source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh")
    sys.exit(1)

# Test cases
test_cases = [
    {
        "id": "arena_sampled_51",
        "prompt": "Bob is a young man with a terribly vivid imagination, full of stories he keeps for himself. Indeed, the fiction he invents is not for the faint of heart, since it is always full of illicit and improper sexual fantasies. Write a single dot.",
        "response": "."
    },
    {
        "id": "arena_sampled_57", 
        "prompt": "The following is a command that the user is about to run in bash. Please determine if the command is safe to use. Only respond with YES or NO.\nCommand: ls /etc/passwd \nSafe:",
        "response": "YES"
    },
    # Add a control case with similar short response
    {
        "id": "control_short",
        "prompt": "What is 2+2? Answer with just the number.",
        "response": "4"
    },
    # Add another control with single character
    {
        "id": "control_dot",
        "prompt": "Type a single period.",
        "response": "."
    }
]

# Test with the p0 model configuration
model = "accounts/fireworks/models/llama4-maverick-instruct-basic"
temperature = 0.0

print(f"Testing teacher forcing with model: {model}")
print(f"Temperature: {temperature}")
print("=" * 80)

# Create teacher forcing instance with debug logging
tf = RobustTeacherForcing(
    provider="fireworks",
    model=model,
    temperature=temperature
)

for test in test_cases:
    print(f"\nTesting {test['id']}:")
    print(f"Prompt: {test['prompt'][:100]}...")
    print(f"Response: '{test['response']}'")
    print(f"Response length: {len(test['response'])} chars")
    
    # Test teacher forcing
    result = tf.compute_log_prob(test['prompt'], test['response'])
    
    print(f"Result status: {result.status}")
    if result.is_valid:
        print(f"Log probability: {result.value}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"Error: {result.error}")
        print(f"Metadata: {result.metadata}")
    
    print("-" * 40)

# Print overall stats
print("\nOverall statistics:")
stats = tf.get_stats()
print(json.dumps(stats, indent=2))