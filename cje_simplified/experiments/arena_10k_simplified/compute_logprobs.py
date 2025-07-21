#!/usr/bin/env python3
"""
Compute log probabilities for responses using teacher forcing.

This uses the Fireworks API to compute log P(response|prompt) for each
response under different models, properly handling chat templates and system prompts.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje_simplified import compute_chat_logprob, Llama4TemplateConfig
from policy_config import POLICIES, get_policy_config


def compute_logprobs_for_responses(
    base_responses_file: str,
    output_file: str,
    max_samples: Optional[int] = None,
    policy_name: str = "base",
) -> List[Dict]:
    """Compute log probabilities for BASE policy responses under a given policy's model.

    This is used for CJE - we always compute log P(base_response | prompt) under
    different policy models to estimate importance weights.

    Args:
        base_responses_file: Path to BASE policy responses JSONL file
        output_file: Where to save log probabilities
        max_samples: Limit number of samples to process
        policy_name: Name of the policy whose model to use for computing log probs
    """

    # Check API key
    if not os.getenv("FIREWORKS_API_KEY"):
        raise ValueError("FIREWORKS_API_KEY environment variable required")

    # Get policy configuration
    policy_config = get_policy_config(policy_name)
    model = policy_config["model"]
    temperature = policy_config["temperature"]
    system_prompt = policy_config["system_prompt"]

    # Load BASE policy responses
    responses = []
    with open(base_responses_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("response"):  # Skip failed responses
                responses.append(data)

    if max_samples:
        responses = responses[:max_samples]

    print(f"Computing log probs for {len(responses)} BASE responses...")
    print(f"Using {policy_name} policy model: {model}")
    print(f"Temperature: {temperature}, System prompt: {system_prompt[:50]}...")

    # Create template config (using Llama4 template for llama4-maverick-instruct-basic)
    template_config = Llama4TemplateConfig()

    # Compute log probabilities
    results: List[Dict[str, Any]] = []
    for i, data in enumerate(responses):
        prompt = data["prompt"]
        response = data["response"]

        # Create chat format with system prompt
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        # Compute log probability using chat format
        result = compute_chat_logprob(
            chat=chat,
            model=model,
            temperature=temperature,
            template_config=template_config,
        )

        # Store result
        output_data = {
            "prompt_id": data["prompt_id"],
            "prompt": prompt,
            "response": response,
            "source_policy": data["policy"],
            "eval_model": model,  # Model from policy config
            "logprob": result.value if result.is_valid else None,
            "error": result.error if not result.is_valid else None,
        }
        results.append(output_data)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(responses)}...")

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    # Print summary
    valid = sum(1 for r in results if r["logprob"] is not None)
    failed = len(results) - valid

    print(f"✓ Computed {valid}/{len(results)} valid log probabilities")
    if failed > 0:
        print(f"⚠️  WARNING: {failed} log prob computations failed!")
        # Show first few errors
        errors = [r for r in results if r["logprob"] is None and r.get("error")][:3]
        for err in errors:
            print(
                f"   - Prompt {err['prompt_id']}: {err.get('error', 'Unknown error')}"
            )
        if failed > 3:
            print(f"   ... and {failed - 3} more failures")
    print(f"✓ Saved to {output_path}")

    return results


def main() -> None:
    """Compute log probabilities for BASE responses under all policy models."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--responses-dir",
        default="data/responses",
        help="Directory with response files",
    )
    parser.add_argument(
        "--output-dir", default="data/logprobs", help="Output directory"
    )
    parser.add_argument("--max-samples", type=int, help="Limit samples per file")

    args = parser.parse_args()

    # Get base responses file
    responses_dir = Path(args.responses_dir)
    base_responses_file = responses_dir / "base_responses.jsonl"

    if not base_responses_file.exists():
        raise FileNotFoundError(
            f"Base responses file not found: {base_responses_file}\n"
            "Please run generate_responses.py first."
        )

    # Process BASE responses under each policy's model
    for policy_name in POLICIES:
        output_file = f"{args.output_dir}/{policy_name}_logprobs.jsonl"

        print(
            f"\nComputing log probs for BASE responses under {policy_name} policy model..."
        )
        compute_logprobs_for_responses(
            base_responses_file=str(base_responses_file),
            output_file=output_file,
            max_samples=args.max_samples,
            policy_name=policy_name,
        )


if __name__ == "__main__":
    main()
