#!/usr/bin/env python3
"""
Temporary stub for log probability computation.
Creates dummy log probabilities to allow pipeline to continue.
"""

import json
from pathlib import Path


def main():
    print("Step 2b: Compute Log Probabilities (STUB)")
    print("⚠️  Using stub implementation due to llama.cpp issues")

    # Load responses
    with open("data/all_responses.jsonl") as f:
        responses = [json.loads(line) for line in f]

    # Create dummy logprobs with reasonable values
    results = []
    for item in responses:
        result = {
            "prompt_id": item["prompt_id"],
            "prompt": item["prompt"],
            "p0_response": item["responses"]["p0"]["response"],
            "logprobs": {
                "p0": -10.0,  # Dummy values
                "pi_clone": -10.0,  # Should be same as p0
                "pi_bad": -30.0,  # Should be much lower
            },
        }
        results.append(result)

    # Save results
    with open("data/logprobs.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"✅ Created dummy log probs for {len(results)} prompts")
    print(
        "⚠️  These are placeholder values - real teacher forcing needed for accurate results"
    )


if __name__ == "__main__":
    main()
