#!/usr/bin/env python3
"""Debug teacher forcing failures with detailed logging."""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from cje.utils.teacher_forcing import RobustTeacherForcing
import openai

# Set up API key
if not os.getenv("FIREWORKS_API_KEY"):
    print("Error: FIREWORKS_API_KEY not set")
    sys.exit(1)

# Test one failing case in detail
test_case = {
    "id": "arena_sampled_51",
    "prompt": "Bob is a young man with a terribly vivid imagination, full of stories he keeps for himself. Indeed, the fiction he invents is not for the faint of heart, since it is always full of illicit and improper sexual fantasies. Write a single dot.",
    "response": "."
}

model = "accounts/fireworks/models/llama4-maverick-instruct-basic"
api_key = os.getenv("FIREWORKS_API_KEY")

print(f"Testing: {test_case['id']}")
print(f"Prompt: {test_case['prompt']}")
print(f"Response: '{test_case['response']}'")
print("=" * 80)

# Test the API calls directly
client = openai.OpenAI(
    api_key=api_key, 
    base_url="https://api.fireworks.ai/inference/v1"
)

# Test 1: Get full text completion
full_text = test_case['prompt'] + test_case['response']
print("\nTest 1: Full text completion")
print(f"Full text: {full_text}")

try:
    full_completion = client.completions.create(
        model=model,
        prompt=full_text,
        max_tokens=0,
        echo=True,
        logprobs=1,
        temperature=0.0
    )
    
    print(f"Success! Got {len(full_completion.choices[0].logprobs.tokens)} tokens")
    print(f"Tokens: {full_completion.choices[0].logprobs.tokens}")
    print(f"Full logprob sum: {sum(lp for lp in full_completion.choices[0].logprobs.token_logprobs if lp is not None)}")
except Exception as e:
    print(f"Failed: {e}")
    print(f"Error type: {type(e)}")
    if hasattr(e, 'response'):
        print(f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'N/A'}")
        print(f"Response text: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")

# Test 2: Get prompt completion
print("\nTest 2: Prompt completion")
print(f"Prompt: {test_case['prompt']}")

try:
    prompt_completion = client.completions.create(
        model=model,
        prompt=test_case['prompt'],
        max_tokens=0,
        echo=True,
        logprobs=1,
        temperature=0.0
    )
    
    print(f"Success! Got {len(prompt_completion.choices[0].logprobs.tokens)} tokens")
    print(f"Prompt logprob sum: {sum(lp for lp in prompt_completion.choices[0].logprobs.token_logprobs if lp is not None)}")
except Exception as e:
    print(f"Failed: {e}")
    print(f"Error type: {type(e)}")

# Test 3: Try a working example for comparison
print("\nTest 3: Working example for comparison")
working_prompt = "What is 2+2? Answer with just the number."
working_response = "4"
working_full = working_prompt + working_response

try:
    working_completion = client.completions.create(
        model=model,
        prompt=working_full,
        max_tokens=0,
        echo=True,
        logprobs=1,
        temperature=0.0
    )
    
    print(f"Success! Got {len(working_completion.choices[0].logprobs.tokens)} tokens")
    print(f"Working example logprob sum: {sum(lp for lp in working_completion.choices[0].logprobs.token_logprobs if lp is not None)}")
except Exception as e:
    print(f"Failed: {e}")

# Test 4: Try generating a response instead of teacher forcing
print("\nTest 4: Try generating a response")
try:
    gen_completion = client.completions.create(
        model=model,
        prompt=test_case['prompt'],
        max_tokens=5,
        temperature=0.0
    )
    
    print(f"Generated text: '{gen_completion.choices[0].text}'")
except Exception as e:
    print(f"Failed: {e}")