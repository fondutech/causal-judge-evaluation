#!/usr/bin/env python3
"""
Regenerate failed responses from existing response files.

This script:
1. Scans existing response files for null responses (failures)
2. Attempts to regenerate only those failed responses
3. Updates the files in-place with successful regenerations
4. Provides detailed statistics on the recovery process
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from pipeline_steps.generate_responses import generate_responses
from policy_config import get_all_policies


def analyze_response_file(file_path: str) -> Tuple[int, int, List[str]]:
    """Analyze a response file for failures.

    Args:
        file_path: Path to response JSONL file

    Returns:
        Tuple of (total_count, failed_count, failed_prompt_ids)
    """
    total = 0
    failed = 0
    failed_ids = []

    if not Path(file_path).exists():
        return 0, 0, []

    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                total += 1
                if data.get("response") is None:
                    failed += 1
                    failed_ids.append(data.get("prompt_id", f"unknown_{total}"))
            except json.JSONDecodeError:
                continue

    return total, failed, failed_ids


def create_filtered_prompts_file(
    original_prompts: str, failed_ids: List[str], output_file: str
) -> int:
    """Create a prompts file containing only the failed prompt IDs.

    Args:
        original_prompts: Path to original prompts.jsonl
        failed_ids: List of prompt IDs that failed
        output_file: Where to save filtered prompts

    Returns:
        Number of prompts written
    """
    failed_set = set(failed_ids)
    count = 0

    with open(original_prompts, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                if data.get("prompt_id") in failed_set:
                    outfile.write(line)
                    count += 1
            except json.JSONDecodeError:
                continue

    return count


def main():
    parser = argparse.ArgumentParser(description="Regenerate failed responses")
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing response files"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        help="Specific policies to regenerate (default: all with failures)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze failures without regenerating",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Maximum retry attempts per request"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Save progress every N responses"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    responses_dir = data_dir / "responses"
    prompts_file = data_dir / "prompts.jsonl"

    if not prompts_file.exists():
        print(f"❌ Prompts file not found: {prompts_file}")
        return 1

    # Get all policies
    all_policies = get_all_policies()
    policy_map = {p["name"]: p for p in all_policies}

    # Analyze all response files
    print("=" * 80)
    print("ANALYZING RESPONSE FILES FOR FAILURES")
    print("=" * 80)

    failure_summary = {}
    total_failures = 0

    for policy in all_policies:
        policy_name = policy["name"]
        response_file = responses_dir / f"{policy_name}_responses.jsonl"

        if not response_file.exists():
            print(f"⚠️  {policy_name}: File not found")
            continue

        total, failed, failed_ids = analyze_response_file(str(response_file))
        failure_summary[policy_name] = {
            "total": total,
            "failed": failed,
            "failed_ids": failed_ids,
            "file": response_file,
        }
        total_failures += failed

        if failed > 0:
            print(
                f"❌ {policy_name}: {failed}/{total} failed ({failed/total*100:.1f}%)"
            )
            # Show first few failed IDs
            sample_ids = failed_ids[:3]
            if len(failed_ids) > 3:
                print(f"   Failed IDs (first 3): {', '.join(sample_ids)}...")
            else:
                print(f"   Failed IDs: {', '.join(sample_ids)}")
        else:
            print(f"✅ {policy_name}: {total}/{total} successful (100%)")

    if total_failures == 0:
        print("\n✨ No failures found! All responses are complete.")
        return 0

    print(f"\n📊 Total failures across all policies: {total_failures}")

    if args.dry_run:
        print(
            "\n🔍 Dry run complete. Use without --dry-run to regenerate failed responses."
        )
        return 0

    # Filter policies to regenerate
    if args.policies:
        policies_to_regen = [
            p
            for p in args.policies
            if p in failure_summary and failure_summary[p]["failed"] > 0
        ]
    else:
        policies_to_regen = [
            p for p in failure_summary if failure_summary[p]["failed"] > 0
        ]

    if not policies_to_regen:
        print("\n✨ No policies need regeneration.")
        return 0

    print("\n" + "=" * 80)
    print("REGENERATING FAILED RESPONSES")
    print("=" * 80)

    overall_stats = {"attempted": 0, "recovered": 0, "still_failed": 0}

    for policy_name in policies_to_regen:
        if policy_name not in policy_map:
            print(f"⚠️  Unknown policy: {policy_name}")
            continue

        policy = policy_map[policy_name]
        failure_info = failure_summary[policy_name]

        print(f"\n🔄 Regenerating {policy_name} ({failure_info['failed']} failures)...")

        # The new generate_responses function will automatically retry failed responses
        # when batch_size is set and skip_failed is False
        results = generate_responses(
            prompts_file=str(prompts_file),
            output_file=str(failure_info["file"]),
            model=policy["model"],
            temperature=policy["temperature"],
            policy_name=policy["name"],
            system_prompt=policy["system_prompt"],
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            retry_delay=1.0,
            max_retry_delay=60.0,
            skip_failed=False,  # Important: Don't skip failed responses
        )

        # Re-analyze to see how many were recovered
        new_total, new_failed, _ = analyze_response_file(str(failure_info["file"]))
        recovered = failure_info["failed"] - new_failed

        overall_stats["attempted"] += failure_info["failed"]
        overall_stats["recovered"] += recovered
        overall_stats["still_failed"] += new_failed

        if recovered > 0:
            print(
                f"  ✅ Recovered: {recovered}/{failure_info['failed']} "
                f"({recovered/failure_info['failed']*100:.1f}%)"
            )
        if new_failed > 0:
            print(f"  ❌ Still failed: {new_failed}")

    # Final summary
    print("\n" + "=" * 80)
    print("REGENERATION SUMMARY")
    print("=" * 80)

    if overall_stats["attempted"] > 0:
        recovery_rate = overall_stats["recovered"] / overall_stats["attempted"] * 100
        print(
            f"📊 Overall recovery: {overall_stats['recovered']}/{overall_stats['attempted']} "
            f"({recovery_rate:.1f}%)"
        )

        if overall_stats["still_failed"] > 0:
            print(f"⚠️  Still failed: {overall_stats['still_failed']} responses")
            print("\nThese may be due to:")
            print("  - Content policy violations")
            print("  - Persistent API issues")
            print("  - Model-specific limitations")
            print("\nConsider:")
            print("  1. Running again with higher --max-retries")
            print("  2. Checking API status and quotas")
            print("  3. Manually reviewing problematic prompts")
        else:
            print("✨ All failures successfully recovered!")

    return 0 if overall_stats["still_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
