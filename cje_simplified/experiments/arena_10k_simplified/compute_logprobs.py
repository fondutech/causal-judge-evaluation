#!/usr/bin/env python3
"""
Compute log probabilities for responses using teacher forcing.

This uses the Fireworks API to compute log P(response|prompt) for each
response under different models.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje_simplified import compute_teacher_forced_logprob


def compute_logprobs_for_responses(
    responses_file: str,
    output_file: str,
    model: str,
    temperature: float = 1.0,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Compute log probabilities for responses."""

    # Check API key
    if not os.getenv("FIREWORKS_API_KEY"):
        raise ValueError("FIREWORKS_API_KEY environment variable required")

    # Load responses
    responses = []
    with open(responses_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("response"):  # Skip failed responses
                responses.append(data)

    if max_samples:
        responses = responses[:max_samples]

    print(f"Computing log probs for {len(responses)} responses...")
    print(f"Model: {model}, Temperature: {temperature}")

    # Compute log probabilities
    results = []
    for i, data in enumerate(responses):
        prompt = data["prompt"]
        response = data["response"]

        # Compute log probability
        result = compute_teacher_forced_logprob(
            prompt=prompt, response=response, model=model, temperature=temperature
        )

        # Store result
        output_data = {
            "prompt_id": data["prompt_id"],
            "prompt": prompt,
            "response": response,
            "source_policy": data["policy"],
            "eval_model": model.split("/")[-1],  # Short name
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
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Print summary
    valid = sum(1 for r in results if r["logprob"] is not None)
    print(f"✓ Computed {valid}/{len(results)} valid log probabilities")
    print(f"✓ Saved to {output_path}")

    return results


def main():
    """Compute log probabilities for all response files."""
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
    parser.add_argument(
        "--model",
        default="accounts/fireworks/models/llama-v3p2-3b-instruct",
        help="Fireworks model for evaluation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for log prob computation",
    )
    parser.add_argument("--max-samples", type=int, help="Limit samples per file")

    args = parser.parse_args()

    # Process all response files
    responses_dir = Path(args.responses_dir)
    for response_file in responses_dir.glob("*_responses.jsonl"):
        policy_name = response_file.stem.replace("_responses", "")
        output_file = f"{args.output_dir}/{policy_name}_logprobs.jsonl"

        print(f"\nProcessing {policy_name} policy...")
        compute_logprobs_for_responses(
            responses_file=str(response_file),
            output_file=output_file,
            model=args.model,
            temperature=args.temperature,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
