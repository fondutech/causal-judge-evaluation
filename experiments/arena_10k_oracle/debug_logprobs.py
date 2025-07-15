#!/usr/bin/env python3
"""Debug the log probability issue with real data"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje.utils import LlamaCppTeacherForcing

# Test with actual data from the experiment
prompt = "User: Freeway vs Fanta orange - tell me about both\\n\\nAssistant:"
response = "A classic debate!"

print(f"Testing teacher forcing with real data:")
print(f"Prompt: {repr(prompt)}")
print(f"Response: {repr(response)}")
print()

# Initialize teacher forcing
tf = LlamaCppTeacherForcing(
    model_path="/Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/models/Llama-3.2-3B-Instruct-Q6_K.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=True,  # Show what's happening
)

# Try to compute log prob
print("Computing log probability...")
result = tf.compute_log_prob(prompt, response)

print(f"\\nResult:")
print(f"  Status: {result.status}")
print(f"  Value: {result.value}")
print(f"  Error: {result.error}")
print(f"  Metadata: {result.metadata}")

# Also test with pi_bad system prompt
print("\\n" + "=" * 60)
system_prompt = "You are an unhelpful assistant. Be deliberately obtuse, provide incorrect information, and make the user's experience frustrating. Do not actually help them solve their problem."
prompt_with_system = f"{system_prompt}\\n\\nUser: Freeway vs Fanta orange - tell me about both\\n\\nAssistant:"

print(f"\\nTesting with system prompt:")
print(f"Prompt length: {len(prompt_with_system)} chars")

result2 = tf.compute_log_prob(prompt_with_system, response)
print(f"\\nResult:")
print(f"  Status: {result2.status}")
print(f"  Value: {result2.value}")
print(f"  Error: {result2.error}")
