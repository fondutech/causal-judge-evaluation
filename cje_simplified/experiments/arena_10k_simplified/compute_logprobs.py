#!/usr/bin/env python3
"""
Compute log probabilities for responses using teacher forcing.

This uses the Fireworks API to compute log P(response|prompt) for each
response under different models, properly handling chat templates and system prompts.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje_simplified import compute_chat_logprob, Llama4TemplateConfig
from policy_config import POLICIES, get_policy_config


def compute_median_logprob(
    chat: List[Dict[str, str]],
    model: str,
    temperature: float,
    template_config: Llama4TemplateConfig,
    num_samples: int = 3,
) -> tuple[Optional[float], List[Dict[str, Any]]]:
    """Compute median logprob from multiple API calls.

    Args:
        chat: Chat messages
        model: Model name
        temperature: Temperature for sampling
        template_config: Template configuration
        num_samples: Number of API calls to make (default: 3)

    Returns:
        Tuple of (median_logprob, sample_history)
        Returns (None, sample_history) if all attempts fail
    """
    valid_logprobs = []
    sample_history = []

    for i in range(num_samples):
        attempt_start = time.time()

        try:
            result = compute_chat_logprob(
                chat=chat,
                model=model,
                temperature=temperature,
                template_config=template_config,
            )

            sample_result: Dict[str, Any] = {
                "sample": i + 1,
                "timestamp": attempt_start,
                "duration": time.time() - attempt_start,
                "success": False,
                "logprob": None,
                "error": None,
            }

            if result.is_valid and result.value is not None:
                # Check for positive logprobs (likely an error)
                if result.value > 0:
                    sample_result["error"] = f"Positive logprob: {result.value}"
                    sample_result["logprob"] = float(result.value)
                else:
                    # Valid negative logprob
                    sample_result["success"] = True
                    sample_result["logprob"] = float(result.value)
                    valid_logprobs.append(result.value)
            else:
                sample_result["error"] = result.error or "Unknown error"

            sample_history.append(sample_result)

        except Exception as e:
            sample_result = {
                "sample": i + 1,
                "timestamp": attempt_start,
                "duration": time.time() - attempt_start,
                "success": False,
                "logprob": None,
                "error": str(e),
            }
            sample_history.append(sample_result)

        # Brief pause between API calls
        if i < num_samples - 1:
            time.sleep(0.2)

    # Calculate median if we have valid values
    if valid_logprobs:
        median_value = statistics.median(valid_logprobs)
        return median_value, sample_history
    else:
        return None, sample_history


def compute_logprobs_for_responses(
    base_responses_file: str,
    output_file: str,
    max_samples: Optional[int] = None,
    policy_name: str = "base",
    max_retries: int = 3,
    num_median_samples: int = 3,
) -> List[Dict]:
    """Compute log probabilities for BASE policy responses under a given policy's model.

    This is used for CJE - we always compute log P(base_response | prompt) under
    different policy models to estimate importance weights.

    Args:
        base_responses_file: Path to BASE policy responses JSONL file
        output_file: Where to save log probabilities
        max_samples: Limit number of samples to process
        policy_name: Name of the policy whose model to use for computing log probs
        max_retries: Maximum number of retries for failed computations (default: 3)
        num_median_samples: Number of samples for median computation (default: 3)

    Returns:
        List of dictionaries with log probability results
    """
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
    print(f"Computing median of {num_median_samples} samples per prompt")

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

        # Compute median logprob with retries
        retry_count = 0
        final_logprob = None
        last_error = None
        all_attempts: List[Dict[str, Any]] = []  # Track all attempts including retries

        while retry_count < max_retries:
            # Get median from multiple samples
            median_logprob, sample_history = compute_median_logprob(
                chat=chat,
                model=model,
                temperature=temperature,
                template_config=template_config,
                num_samples=num_median_samples,
            )

            # Record this retry attempt
            retry_attempt = {
                "retry": retry_count + 1,
                "median_logprob": median_logprob,
                "samples": sample_history,
                "valid_samples": sum(1 for s in sample_history if s["success"]),
            }
            all_attempts.append(retry_attempt)

            if median_logprob is not None:
                # Success!
                final_logprob = median_logprob
                print(
                    f"  [{i + 1}/{len(responses)}] {data['prompt_id']}: "
                    f"median={median_logprob:.3f} "
                    f"(from {retry_attempt['valid_samples']} valid samples)"
                )
                break
            else:
                # All samples failed
                errors = [s["error"] for s in sample_history if s["error"]]
                last_error = "; ".join(set(errors))  # Unique errors
                retry_count += 1

                if retry_count < max_retries:
                    print(
                        f"  Retry {retry_count}/{max_retries} for {data['prompt_id']}: "
                        f"all {num_median_samples} samples failed"
                    )
                    time.sleep(1)  # Longer pause before retry

        # Record result
        result = {
            "prompt_id": data["prompt_id"],
            "prompt": prompt,
            "response": response,
            "source_policy": "base",  # Always computing base responses
            "eval_model": model,
            "logprob": final_logprob,
            "error": last_error if final_logprob is None else None,
            "retries": retry_count if retry_count > 0 else None,
            "attempt_history": all_attempts if all_attempts else None,
        }
        results.append(result)

        if final_logprob is None:
            print(f"  ❌ Failed after {retry_count} retries: {last_error}")

    # Save results
    print(f"\nSaving results to {output_file}")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Print summary statistics
    successful = sum(1 for r in results if r["logprob"] is not None)
    print(f"\nSummary:")
    print(f"  Successful: {successful}/{len(results)}")

    if successful > 0:
        logprobs = [r["logprob"] for r in results if r["logprob"] is not None]
        print(f"  Mean logprob: {sum(logprobs) / len(logprobs):.3f}")
        print(f"  Min logprob: {min(logprobs):.3f}")
        print(f"  Max logprob: {max(logprobs):.3f}")

    # Analyze retries
    retried = [r for r in results if r.get("retries")]
    if retried:
        print(f"  Samples requiring retries: {len(retried)}")

    # Analyze sample variance
    print("\nSample variance analysis:")
    for result in results[:3]:  # Show first 3 for brevity
        if result.get("attempt_history"):
            prompt_id = result["prompt_id"]
            # Get all successful logprobs from all attempts
            all_logprobs = []
            for attempt in result["attempt_history"]:
                for sample in attempt["samples"]:
                    if sample["success"]:
                        all_logprobs.append(sample["logprob"])

            if len(all_logprobs) > 1:
                variance = statistics.stdev(all_logprobs)
                print(
                    f"  {prompt_id}: range=[{min(all_logprobs):.2f}, {max(all_logprobs):.2f}], "
                    f"stdev={variance:.2f}, median={result['logprob']:.2f}"
                )

    return results


def main() -> None:
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute log probabilities for responses"
    )
    parser.add_argument(
        "--responses-dir",
        required=True,
        help="Directory containing response JSONL files",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save log probability files"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["base", "clone", "unhelpful"],
        help="List of policies to compute log probs for (default: all)",
    )
    parser.add_argument(
        "--num-median-samples",
        type=int,
        default=3,
        help="Number of samples for median computation (default: 3)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each policy
    base_responses_file = Path(args.responses_dir) / "base_responses.jsonl"
    if not base_responses_file.exists():
        raise FileNotFoundError(f"Base responses file not found: {base_responses_file}")

    for policy in args.policies:
        print(f"\n{'=' * 60}")
        print(f"Computing log probs for {policy} policy")
        print(f"{'=' * 60}")

        output_file = output_dir / f"{policy}_logprobs.jsonl"
        compute_logprobs_for_responses(
            base_responses_file=str(base_responses_file),
            output_file=str(output_file),
            max_samples=args.max_samples,
            policy_name=policy,
            num_median_samples=args.num_median_samples,
        )


if __name__ == "__main__":
    main()
