#!/usr/bin/env python3
"""
Generate responses for Arena prompts using different policies.

This simulates having multiple policies (base, improved, etc.) by using
different models or parameters.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import time

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def generate_responses(
    prompts_file: str,
    output_file: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_responses: Optional[int] = None,
    policy_name: str = "base",
) -> List[Dict]:
    """Generate responses for prompts using specified model/parameters."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    import openai

    client = openai.OpenAI(api_key=api_key)

    # Load prompts
    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    if max_responses:
        prompts = prompts[:max_responses]

    print(f"Generating {len(prompts)} responses with {policy_name} policy...")
    print(f"Model: {model}, Temperature: {temperature}")

    # Generate responses
    results = []
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]

        try:
            # Call API
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200,  # Keep responses short for cost
            )

            result = {
                "prompt_id": prompt_data["prompt_id"],
                "prompt": prompt,
                "response": response.choices[0].message.content,
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


def main():
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

    # Define policies (different model/temperature combinations)
    policies = [
        {"name": "base", "model": "gpt-3.5-turbo", "temperature": 0.7},
        {"name": "creative", "model": "gpt-3.5-turbo", "temperature": 1.2},
        {"name": "focused", "model": "gpt-3.5-turbo", "temperature": 0.3},
    ]

    for policy in policies:
        output_file = f"{args.output_dir}/{policy['name']}_responses.jsonl"
        generate_responses(
            prompts_file=args.prompts,
            output_file=output_file,
            model=policy["model"],
            temperature=policy["temperature"],
            policy_name=policy["name"],
            max_responses=args.max_responses,
        )


if __name__ == "__main__":
    main()
