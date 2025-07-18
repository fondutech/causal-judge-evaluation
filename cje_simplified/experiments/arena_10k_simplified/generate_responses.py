#!/usr/bin/env python3
"""
Generate responses for Arena prompts using different policies.

Uses Fireworks API with different system prompts to simulate different policies:
- base/clone: Helpful assistant
- unhelpful: Deliberately confusing assistant
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import requests  # type: ignore

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def generate_responses(
    prompts_file: str,
    output_file: str,
    model: str,
    temperature: float = 0.7,
    max_responses: Optional[int] = None,
    policy_name: str = "base",
    system_prompt: str = "You are a helpful assistant.",
) -> List[Dict]:
    """Generate responses for prompts using Fireworks API.

    Args:
        prompts_file: Path to JSONL file with prompts
        output_file: Where to save responses
        model: Fireworks model identifier
        temperature: Sampling temperature
        max_responses: Limit number of responses to generate
        policy_name: Name of the policy (for tracking)
        system_prompt: System prompt to use

    Returns:
        List of response dictionaries
    """
    # Check for API key
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable required")

    # Load prompts
    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    if max_responses:
        prompts = prompts[:max_responses]

    print(f"Generating {len(prompts)} responses with {policy_name} policy...")
    print(f"Model: {model}, Temperature: {temperature}")
    print(f"System prompt: {system_prompt[:50]}...")

    # Fireworks API endpoint
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Generate responses
    results = []
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]

        try:
            # Prepare messages with system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Call Fireworks API
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 200,  # Keep responses short for cost
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            response_data = response.json()

            result = {
                "prompt_id": prompt_data["prompt_id"],
                "prompt": prompt,
                "response": response_data["choices"][0]["message"]["content"],
                "policy": policy_name,
                "model": model,
                "temperature": temperature,
            }
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{len(prompts)} responses...")

        except Exception as e:
            print(f"Error on prompt {i}: {e}")
            # Add failed result
            result = {
                "prompt_id": prompt_data["prompt_id"],
                "prompt": prompt,
                "response": None,
                "policy": policy_name,
                "error": str(e),
            }
            results.append(result)

        # Rate limiting
        time.sleep(0.1)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"✓ Saved {len(results)} responses to {output_path}")
    return results


def main() -> None:
    """Generate responses for different policies."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts", default="data/arena_prompts.jsonl", help="Input prompts file"
    )
    parser.add_argument(
        "--output-dir", default="data/responses", help="Output directory"
    )
    parser.add_argument("--max-responses", type=int, help="Limit number of responses")

    args = parser.parse_args()

    # Define policies with different system prompts
    # Using the same model for all to isolate the effect of system prompts
    model = "accounts/fireworks/models/llama-v3p1-8b-instruct"

    policies: List[Dict[str, Any]] = [
        {
            "name": "base",
            "model": model,
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
        },
        {
            "name": "clone",
            "model": model,
            "temperature": 0.7,
            "system_prompt": "You are a helpful assistant.",
        },
        {
            "name": "unhelpful",
            "model": model,
            "temperature": 0.7,
            "system_prompt": "You are an unhelpful assistant that deliberately confuses and misleads the user.",
        },
    ]

    for policy in policies:
        output_file = f"{args.output_dir}/{policy['name']}_responses.jsonl"
        generate_responses(
            prompts_file=args.prompts,
            output_file=output_file,
            model=policy["model"],
            temperature=policy["temperature"],
            policy_name=policy["name"],
            system_prompt=policy["system_prompt"],
            max_responses=args.max_responses,
        )


if __name__ == "__main__":
    main()
