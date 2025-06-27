#!/usr/bin/env python3
"""
Re-score P0 samples that have log probability = -50.0.

This script:
1. Loads the 140 failed samples
2. Re-computes their log probabilities using the same P0 configuration
3. Saves the results with proper error handling
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.progress import track

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner

console = Console()


def load_samples_to_rescore() -> List[Dict[str, Any]]:
    """Load the samples that need re-scoring."""
    input_file = Path("../data/p0_samples_to_rescore.jsonl")

    if not input_file.exists():
        console.print("[red]❌ File not found: p0_samples_to_rescore.jsonl[/red]")
        console.print("Run extract_failed_p0_samples.py first!")
        return []

    samples = []
    with open(input_file, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    return samples


def rescore_p0_batch(
    batch: List[Dict[str, Any]], runner: APIPolicyRunner
) -> List[Dict[str, Any]]:
    """Re-score a batch of P0 samples with proper error handling."""
    results = []

    for item in batch:
        prompt_id = item["prompt_id"]
        context = item["prompt"]
        response = item["response"]

        result = {
            "prompt_id": prompt_id,
            "prompt": context,
            "response": response,
            "metadata": item.get("metadata", {}),
            "original_logprob": item["original_total_logprob"],
        }

        try:
            # Compute log probability
            logp = runner.log_prob(context, response)

            # Validate the result
            if not isinstance(logp, (int, float)):
                raise ValueError(f"Invalid log prob type: {type(logp)}")
            if logp > 0:
                raise ValueError(f"Positive log prob: {logp}")
            if logp == 0.0:
                raise ValueError("Exactly 0.0 log prob (suspicious)")

            result["total_logprob"] = float(logp)
            result["rescore_status"] = "success"

        except Exception as e:
            # Track failure explicitly
            result["total_logprob"] = None
            result["rescore_status"] = "failed"
            result["rescore_error"] = str(e)
            result["error_type"] = type(e).__name__

            console.print(f"[red]❌ Failed to rescore {prompt_id}: {e}[/red]")

        results.append(result)

    return results


def main():
    """Re-score failed P0 samples."""
    console.print("[bold blue]🔄 Re-scoring P0 Failures[/bold blue]\n")

    # Load samples
    samples = load_samples_to_rescore()
    if not samples:
        return

    console.print(f"📂 Loaded {len(samples)} samples to re-score")

    # Initialize P0 runner with exact same configuration
    console.print("\n🔧 Initializing P0 runner...")
    runner = APIPolicyRunner(
        provider="fireworks",
        model_name="accounts/fireworks/models/llama-v3p2-3b-instruct",
        temperature=0.5,
        max_new_tokens=512,
        batch_size=1,  # Process one at a time for better error handling
        completions_template_format="llama4",
    )

    # Process in batches
    batch_size = 10
    all_results = []
    successes = 0
    failures = 0

    for i in track(
        range(0, len(samples), batch_size), description="Re-scoring batches"
    ):
        batch = samples[i : i + batch_size]
        results = rescore_p0_batch(batch, runner)

        # Count successes/failures
        for result in results:
            if result["rescore_status"] == "success":
                successes += 1
            else:
                failures += 1

        all_results.extend(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"../data/p0_rescored_{timestamp}.jsonl")

    with open(output_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    console.print(f"\n✅ Re-scoring complete!")
    console.print(f"   Successes: {successes}")
    console.print(f"   Failures: {failures}")
    console.print(f"   Saved to: {output_file}")

    # Save failure log if any
    if failures > 0:
        failure_file = Path(f"../data/p0_rescore_failures_{timestamp}.json")
        failed_items = [r for r in all_results if r["rescore_status"] == "failed"]

        with open(failure_file, "w") as f:
            json.dump(
                {
                    "total_failures": failures,
                    "failure_rate": failures / len(samples),
                    "failures": failed_items,
                },
                f,
                indent=2,
            )

        console.print(f"[yellow]⚠️  Failure details saved to: {failure_file}[/yellow]")

    # Show sample results
    console.print("\n[bold]Sample Results:[/bold]")
    for result in all_results[:3]:
        if result["rescore_status"] == "success":
            console.print(f"✅ {result['prompt_id']}: {result['total_logprob']:.2f}")
        else:
            console.print(f"❌ {result['prompt_id']}: {result['rescore_error']}")


if __name__ == "__main__":
    main()
