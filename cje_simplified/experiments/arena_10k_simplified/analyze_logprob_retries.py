#!/usr/bin/env python3
"""
Analyze retry patterns in logprob computation files.

This helps identify systematic issues with specific prompts or models.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import argparse


def analyze_retries(logprobs_dir: str) -> None:
    """Analyze retry patterns across all logprob files."""

    logprobs_path = Path(logprobs_dir)
    if not logprobs_path.exists():
        print(f"❌ Directory not found: {logprobs_dir}")
        return

    # Collect all retry data
    all_retries = []
    total_computations = 0
    total_with_retries = 0
    total_failures = 0

    # Stats per policy
    policy_stats: Dict[str, Dict[str, Any]] = {}

    for file_path in logprobs_path.glob("*_logprobs.jsonl"):
        policy = file_path.stem.replace("_logprobs", "")
        policy_stats[policy] = {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "retried": 0,
            "retry_reasons": defaultdict(int),
        }

        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                total_computations += 1
                policy_stats[policy]["total"] += 1

                if data.get("logprob") is not None:
                    policy_stats[policy]["succeeded"] += 1
                else:
                    total_failures += 1
                    policy_stats[policy]["failed"] += 1

                if data.get("retries") and data["retries"] > 0:
                    total_with_retries += 1
                    policy_stats[policy]["retried"] += 1

                # Analyze attempt history
                if data.get("attempt_history") and len(data["attempt_history"]) > 1:
                    retry_info = {
                        "prompt_id": data["prompt_id"],
                        "policy": policy,
                        "attempts": len(data["attempt_history"]),
                        "final_success": data.get("logprob") is not None,
                        "history": data["attempt_history"],
                    }
                    all_retries.append(retry_info)

                    # Count retry reasons
                    for attempt in data["attempt_history"][:-1]:
                        reason = attempt.get("error", "Unknown").split(":")[0]
                        policy_stats[policy]["retry_reasons"][reason] += 1

    # Print overall summary
    print("📊 Logprob Computation Retry Analysis")
    print("=" * 50)
    print(f"\nOverall Statistics:")
    print(f"  Total computations: {total_computations}")
    print(
        f"  Successful: {total_computations - total_failures} ({100*(total_computations - total_failures)/total_computations:.1f}%)"
    )
    print(f"  Failed: {total_failures} ({100*total_failures/total_computations:.1f}%)")
    print(
        f"  Required retries: {total_with_retries} ({100*total_with_retries/total_computations:.1f}%)"
    )

    # Per-policy stats
    print(f"\n📈 Per-Policy Statistics:")
    for policy, stats in sorted(policy_stats.items()):
        print(f"\n{policy}:")
        print(f"  Total: {stats['total']}")
        print(f"  Success rate: {100*stats['succeeded']/stats['total']:.1f}%")
        if stats["retried"] > 0:
            print(
                f"  Required retries: {stats['retried']} ({100*stats['retried']/stats['total']:.1f}%)"
            )
        if stats["retry_reasons"]:
            print(f"  Retry reasons:")
            for reason, count in sorted(
                stats["retry_reasons"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    - {reason}: {count}")

    # Analyze problematic prompts
    prompt_retry_counts: Dict[str, int] = defaultdict(int)
    prompt_failure_counts: Dict[str, int] = defaultdict(int)

    for retry_info in all_retries:
        prompt_retry_counts[retry_info["prompt_id"]] += 1
        if not retry_info["final_success"]:
            prompt_failure_counts[retry_info["prompt_id"]] += 1

    # Show most problematic prompts
    if prompt_retry_counts:
        print(f"\n🔥 Most Problematic Prompts (by retry frequency):")
        sorted_prompts = sorted(
            prompt_retry_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for prompt_id, count in sorted_prompts:
            failures = prompt_failure_counts.get(prompt_id, 0)
            print(
                f"  {prompt_id}: {count} retries across policies ({failures} final failures)"
            )

    # Analyze retry patterns
    if all_retries:
        print(f"\n🔍 Retry Pattern Analysis:")

        # Group by number of attempts
        attempts_distribution: Dict[int, int] = defaultdict(int)
        for retry_info in all_retries:
            attempts_distribution[retry_info["attempts"]] += 1

        print("\nAttempts distribution:")
        for attempts, count in sorted(attempts_distribution.items()):
            print(f"  {attempts} attempts: {count} cases")

        # Success rate after retries
        retry_successes = sum(1 for r in all_retries if r["final_success"])
        print(
            f"\nSuccess rate after retries: {100*retry_successes/len(all_retries):.1f}%"
        )

        # Most common error patterns
        error_patterns: Dict[str, int] = defaultdict(int)
        for retry_info in all_retries:
            for attempt in retry_info["history"][:-1]:
                if attempt.get("error"):
                    error_patterns[attempt["error"][:50]] += 1

        if error_patterns:
            print("\nMost common error patterns:")
            for error, count in sorted(
                error_patterns.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                print(f"  - {error}: {count}")


def main() -> None:
    """Analyze retry patterns in logprob files."""
    parser = argparse.ArgumentParser(
        description="Analyze retry patterns in logprob computations"
    )
    parser.add_argument(
        "--logprobs-dir",
        default="data/logprobs",
        help="Directory containing logprob files",
    )

    args = parser.parse_args()
    analyze_retries(args.logprobs_dir)


if __name__ == "__main__":
    main()
