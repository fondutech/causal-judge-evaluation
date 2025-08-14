#!/usr/bin/env python3
"""
End-to-end test for the Arena experiment pipeline.

This test runs the ACTUAL production pipeline (generate_arena_data.py)
with a small number of samples to verify functionality.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
import shutil
from typing import Dict, Any


def count_records(file_path: Path, field: str = None) -> int:
    """Count records in a JSONL file."""
    if not file_path.exists():
        return 0

    count = 0
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                if field:
                    data = json.loads(line)
                    if field in data.get("metadata", {}) or field in data:
                        count += 1
                else:
                    count += 1
    return count


def verify_dataset(test_dir: Path, n_samples: int) -> Dict[str, Any]:
    """Verify the generated dataset has expected structure."""
    results: Dict[str, Any] = {"success": True, "errors": []}

    # Check prompts
    prompts_file = test_dir / "prompts.jsonl"
    if not prompts_file.exists():
        results["success"] = False
        results["errors"].append("prompts.jsonl not found")
    else:
        n_prompts = count_records(prompts_file)
        print(f"✓ Prompts: {n_prompts}")
        if n_prompts != n_samples:
            results["errors"].append(f"Expected {n_samples} prompts, got {n_prompts}")

    # Check responses for each policy
    from experiment_config import POLICY_NAMES

    for policy in POLICY_NAMES:
        response_file = test_dir / "responses" / f"{policy}_responses.jsonl"
        if not response_file.exists():
            results["success"] = False
            results["errors"].append(f"{policy}_responses.jsonl not found")
        else:
            n_responses = count_records(response_file)
            n_judge = count_records(response_file, "judge_score")
            n_oracle = count_records(response_file, "oracle_label")
            print(
                f"✓ {policy}: {n_responses} responses, {n_judge} judge scores, {n_oracle} oracle labels"
            )

    # Check log probabilities
    for policy in POLICY_NAMES:
        logprob_file = test_dir / "logprobs" / f"{policy}_logprobs.jsonl"
        if not logprob_file.exists():
            results["success"] = False
            results["errors"].append(f"{policy}_logprobs.jsonl not found")
        else:
            n_logprobs = count_records(logprob_file)
            print(f"✓ {policy} logprobs: {n_logprobs}")

    # Check final dataset
    dataset_file = test_dir / "cje_dataset.jsonl"
    if not dataset_file.exists():
        results["success"] = False
        results["errors"].append("cje_dataset.jsonl not found")
    else:
        n_dataset = count_records(dataset_file)
        print(f"✓ Final dataset: {n_dataset} samples")

        # Check dataset structure
        with open(dataset_file, "r") as f:
            sample = json.loads(f.readline())
            required_fields = [
                "prompt",
                "response",
                "base_policy_logprob",
                "target_policy_logprobs",
                "metadata",
            ]
            for field in required_fields:
                if field not in sample:
                    results["success"] = False
                    results["errors"].append(f"Dataset missing required field: {field}")

            # Check metadata
            if "judge_score" not in sample.get("metadata", {}):
                results["errors"].append("Dataset missing judge_score in metadata")
            if "oracle_label" not in sample.get("metadata", {}):
                results["errors"].append("Dataset missing oracle_label in metadata")

    return results


def main() -> int:
    """Run end-to-end pipeline test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Arena experiment pipeline")
    parser.add_argument(
        "--n-samples", type=int, default=10, help="Number of test samples"
    )
    parser.add_argument(
        "--test-dir", default="test_pipeline_data", help="Test data directory"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean test directory first"
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)

    # Clean if requested
    if args.clean and test_dir.exists():
        print(f"Cleaning {test_dir}...")
        shutil.rmtree(test_dir)

    print("=" * 60)
    print("Testing Arena Experiment Pipeline")
    print("=" * 60)
    print(f"Samples: {args.n_samples}")
    print(f"Test directory: {test_dir}")
    print()

    # Run the actual production pipeline
    cmd = [
        "poetry",
        "run",
        "python",
        "generate_arena_data.py",
        "--data-dir",
        str(test_dir),
        "--n-samples",
        str(args.n_samples),
        "--batch-size",
        "5",  # Small batch for testing
        "--max-tokens",
        "50",  # Short responses for speed
    ]

    print(f"Running: {' '.join(cmd)}")
    print("This will take a few minutes...")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Pipeline failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return 1

    print("Pipeline completed. Verifying output...")
    print()

    # Verify the dataset
    verification = verify_dataset(test_dir, args.n_samples)

    print()
    print("=" * 60)
    if verification["success"] and not verification["errors"]:
        print("✅ All tests passed!")
    else:
        print("❌ Test failures:")
        for error in verification["errors"]:
            print(f"  - {error}")
        return 1

    print("=" * 60)

    # Test resume capability
    print("\n🔄 Testing resume capability...")
    print("Running pipeline again (should skip everything)...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check that it skipped work
    if "Skipping" in result.stdout or "already exist" in result.stdout.lower():
        print("✅ Resume capability working (skipped existing work)")
    else:
        print("⚠️  Pipeline may have redone work")

    # Optional: Test analysis
    print("\n📊 Testing analysis (optional)...")
    analysis_cmd = [
        "poetry",
        "run",
        "python",
        "analyze_dataset.py",
        "--data",
        str(test_dir / "cje_dataset.jsonl"),
        "--estimator",
        "calibrated-ips",
        "--oracle-coverage",
        "0.5",
        "--no-plots",  # Skip plots for speed
        "--quiet",
    ]

    print(f"Running: {' '.join(analysis_cmd)}")
    result = subprocess.run(analysis_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Analysis completed successfully")
    else:
        print("❌ Analysis failed (this may be expected with small sample size)")
        print("Error:", result.stderr[:500])

    print("\n✨ Pipeline test complete!")

    # Ask about cleanup
    if test_dir.exists():
        response = input(f"\nDelete test directory {test_dir}? (y/n): ").strip().lower()
        if response == "y":
            shutil.rmtree(test_dir)
            print("Test directory deleted.")

    return 0


if __name__ == "__main__":
    # Check for API keys
    if not os.getenv("FIREWORKS_API_KEY"):
        print("❌ FIREWORKS_API_KEY not set")
        print("Run: source set_secrets.sh")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set (needed for judge/oracle scoring)")
        print("The pipeline will fail at the scoring step.")

    sys.exit(main())
