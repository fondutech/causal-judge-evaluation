#!/usr/bin/env python3
"""
Example of using llama.cpp for teacher forcing in Arena 10K experiment.

This provides a local, deterministic alternative to the Fireworks API.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje.utils import RobustTeacherForcing


def main() -> None:
    """Example usage of llama.cpp teacher forcing."""

    # Example model paths (adjust to your setup)
    LLAMA3_8B_PATH = "~/models/llama-3-8b-instruct.Q4_K_M.gguf"
    LLAMA3_70B_PATH = "~/models/llama-3-70b-instruct.Q4_K_M.gguf"

    # Test prompt and response
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."

    print("Testing llama.cpp Teacher Forcing")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print()

    # Example 1: Basic usage
    print("1. Basic llama.cpp usage:")
    tf_basic = RobustTeacherForcing(
        provider="llama_cpp",
        model=LLAMA3_8B_PATH,
        temperature=0.5,
        seed=42,
        n_ctx=4096,  # Context window
        n_gpu_layers=-1,  # Use all GPU layers
    )

    result = tf_basic.compute_log_prob(prompt, response)
    if result.is_valid:
        print(f"   Log probability: {result.value:.4f}")
        print(f"   Metadata: {result.metadata}")
    else:
        print(f"   Failed: {result.error}")

    # Example 2: With system prompt
    print("\n2. With system prompt:")
    tf_system = RobustTeacherForcing(
        provider="llama_cpp",
        model=LLAMA3_8B_PATH,
        temperature=0.5,
        seed=42,
        system_prompt="You are a helpful assistant.",
        n_ctx=4096,
    )

    result = tf_system.compute_log_prob(prompt, response)
    if result.is_valid:
        print(f"   Log probability: {result.value:.4f}")
    else:
        print(f"   Failed: {result.error}")

    # Example 3: Multiple policies with different models
    print("\n3. Multiple policies setup (Arena 10K style):")

    from typing import Dict, Any

    policies: Dict[str, Dict[str, Any]] = {
        "p0": {
            "model": LLAMA3_8B_PATH,
            "temperature": 0.5,
        },
        "pi_clone": {
            "model": LLAMA3_8B_PATH,  # Same model as p0
            "temperature": 0.5,
        },
        "pi_bigger_model": {
            "model": LLAMA3_70B_PATH,  # Larger model
            "temperature": 0.5,
        },
        "pi_bad": {
            "model": LLAMA3_8B_PATH,
            "temperature": 1.0,  # Higher temperature
            "system_prompt": "You are an unhelpful assistant.",
        },
    }

    # Create teacher forcing instances
    tf_instances: Dict[str, RobustTeacherForcing] = {}
    for policy_name, config in policies.items():
        tf_instances[policy_name] = RobustTeacherForcing(
            provider="llama_cpp",
            model=str(config["model"]),
            temperature=float(config.get("temperature", 0.5)),
            system_prompt=config.get("system_prompt"),
            seed=42,
            n_ctx=4096,
            n_gpu_layers=-1,
        )

    # Compute log probs for each policy
    print("\n   Computing log probabilities:")
    for policy_name, tf in tf_instances.items():
        result = tf.compute_log_prob(prompt, response)
        if result.is_valid:
            print(f"   {policy_name}: {result.value:.4f}")
        else:
            print(f"   {policy_name}: FAILED - {result.error}")

    # Show advantages
    print("\n" + "=" * 60)
    print("Advantages of llama.cpp:")
    print("1. Fully deterministic (with seed)")
    print("2. No token boundary issues")
    print("3. No API costs or rate limits")
    print("4. Works offline")
    print("5. Supports quantized models (Q4_K_M, Q5_K_M, etc.)")

    print("\nIntegration with Arena 10K:")
    print("In phase1_dataset_preparation/02b_compute_logprobs.py:")
    print(
        """
    tf_instances[policy_name] = RobustTeacherForcing(
        provider="llama_cpp",  # Instead of "fireworks"
        model=model_path,      # Path to GGUF file
        temperature=temperature,
        system_prompt=system_prompt,
        seed=42,
        n_ctx=8192,
        n_gpu_layers=-1,
    )
    """
    )


if __name__ == "__main__":
    main()
