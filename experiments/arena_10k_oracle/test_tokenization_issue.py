#!/usr/bin/env python3
"""Test tokenization boundary issue."""

import os
import sys
from pathlib import Path
import tiktoken

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Use tiktoken to check tokenization
tokenizer = tiktoken.get_encoding("cl100k_base")

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
    {
        "id": "control_works",
        "prompt": "What is 2+2?",
        "response": "4"
    },
    {
        "id": "control_dot_fails",
        "prompt": "Type a single period.",
        "response": "."
    }
]

for test in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing: {test['id']}")
    print(f"Prompt: '{test['prompt'][:50]}...'")
    print(f"Response: '{test['response']}'")
    
    # Tokenize prompt alone
    prompt_tokens = tokenizer.encode(test['prompt'])
    print(f"\nPrompt tokens: {len(prompt_tokens)}")
    print(f"Last 5 prompt tokens: {prompt_tokens[-5:]}")
    
    # Tokenize full text
    full_text = test['prompt'] + test['response']
    full_tokens = tokenizer.encode(full_text)
    print(f"\nFull text tokens: {len(full_tokens)}")
    
    # Check if prompt tokens are a prefix
    is_prefix = (len(full_tokens) >= len(prompt_tokens) and 
                 full_tokens[:len(prompt_tokens)] == prompt_tokens)
    
    print(f"\nIs prompt a prefix? {is_prefix}")
    
    if not is_prefix:
        # Show where they differ
        print("\nTokenization difference detected!")
        print(f"Prompt ends with: {tokenizer.decode(prompt_tokens[-5:])}")
        print(f"Full text at boundary: {tokenizer.decode(full_tokens[len(prompt_tokens)-5:len(prompt_tokens)+5])}")
        
        # Check what happens at the boundary
        boundary_start = max(0, len(prompt_tokens) - 10)
        print(f"\nPrompt tokens around boundary: {prompt_tokens[boundary_start:]}")
        print(f"Full tokens around boundary: {full_tokens[boundary_start:boundary_start+len(prompt_tokens[boundary_start:])+5]}")
        
    # Also check the response tokens
    if is_prefix:
        response_tokens = full_tokens[len(prompt_tokens):]
        print(f"\nResponse tokens: {response_tokens}")
        print(f"Response decoded: '{tokenizer.decode(response_tokens)}'")