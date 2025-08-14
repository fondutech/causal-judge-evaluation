#!/usr/bin/env python3
"""
Test script to verify resume functionality works correctly.

This script simulates interrupted runs and verifies that:
1. Response generation resumes correctly
2. Scoring resumes correctly
3. No duplicate work is done
4. Temp files don't collide
"""

import json
import subprocess
import time
from pathlib import Path
import sys
import os


def count_records(file_path: str, field: str = None) -> int:
    """Count records in a JSONL file, optionally filtering by field presence."""
    if not Path(file_path).exists():
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


def run_with_interrupt(cmd: str, interrupt_after: float = 3.0) -> subprocess.Popen:
    """Start a command and interrupt it after a delay."""
    print(f"Starting: {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(interrupt_after)
    print(f"Interrupting after {interrupt_after} seconds...")
    proc.terminate()
    time.sleep(0.5)  # Give it time to clean up
    if proc.poll() is None:
        proc.kill()
    return proc


def main() -> None:
    """Test resume functionality."""
    test_dir = Path("test_resume_data")

    # Clean up any previous test
    if test_dir.exists():
        import shutil

        shutil.rmtree(test_dir)

    print("=" * 60)
    print("Testing Resume Functionality")
    print("=" * 60)

    # Test 1: Generate prompts and interrupt
    print("\n📝 Test 1: Prompt extraction")
    cmd = f"poetry run python generate_arena_data.py --data-dir {test_dir} --n-samples 50 --batch-size 10"

    # Run and interrupt during prompt extraction
    proc = run_with_interrupt(cmd, interrupt_after=2.0)

    prompts_before = count_records(str(test_dir / "prompts.jsonl"))
    print(f"Prompts after interrupt: {prompts_before}")

    # Resume and complete
    print("Resuming...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    prompts_after = count_records(str(test_dir / "prompts.jsonl"))
    print(f"Prompts after resume: {prompts_after}")

    if prompts_after != 50:
        print(f"❌ Expected 50 prompts, got {prompts_after}")
    else:
        print("✅ Prompt extraction resumed correctly")

    # Test 2: Response generation with interruption
    print("\n📝 Test 2: Response generation")

    # Check initial state
    base_responses = test_dir / "responses" / "base_responses.jsonl"
    if base_responses.exists():
        responses_before = count_records(str(base_responses))
        print(f"Existing responses: {responses_before}")

    # Run full pipeline with small batches
    cmd = f"poetry run python generate_arena_data.py --data-dir {test_dir} --n-samples 50 --batch-size 5"

    # Let it run for a bit to generate some responses
    proc = run_with_interrupt(cmd, interrupt_after=10.0)

    responses_mid = count_records(str(base_responses)) if base_responses.exists() else 0
    print(f"Responses after interrupt: {responses_mid}")

    if responses_mid > 0:
        print("✅ Incremental save worked")

    # Resume and complete
    print("Resuming full pipeline...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    responses_final = (
        count_records(str(base_responses)) if base_responses.exists() else 0
    )
    print(f"Responses after resume: {responses_final}")

    # Test 3: Check scoring resume
    print("\n📝 Test 3: Scoring resume")

    # Count scores before
    scores_before = count_records(str(base_responses), field="judge_score")
    print(f"Judge scores before: {scores_before}")

    # Force rescore to test
    cmd = f"poetry run python pipeline_steps/add_scores_with_resume.py {base_responses} --type judge --batch-size 5"

    # Interrupt during scoring
    proc = run_with_interrupt(cmd, interrupt_after=5.0)

    scores_mid = count_records(str(base_responses), field="judge_score")
    print(f"Judge scores after interrupt: {scores_mid}")

    # Resume scoring
    print("Resuming scoring...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    scores_final = count_records(str(base_responses), field="judge_score")
    print(f"Judge scores after resume: {scores_final}")

    if scores_final == responses_final:
        print("✅ Scoring resumed correctly")
    else:
        print(f"⚠️ Expected {responses_final} scores, got {scores_final}")

    # Test 4: Parallel temp files
    print("\n📝 Test 4: Parallel temp file safety")

    # Start two scoring processes in parallel
    cmd1 = f"poetry run python pipeline_steps/add_scores_with_resume.py {base_responses} --type judge --force &"
    cmd2 = f"poetry run python pipeline_steps/add_scores_with_resume.py {test_dir}/responses/clone_responses.jsonl --type judge &"

    print("Starting parallel scoring processes...")
    subprocess.run(cmd1 + " " + cmd2 + " wait", shell=True)

    # Check for leftover temp files
    temp_files = list(test_dir.glob("**/*.tmp.*"))
    if temp_files:
        print(f"⚠️ Found {len(temp_files)} leftover temp files:")
        for f in temp_files:
            print(f"  - {f}")
    else:
        print("✅ No temp file collisions or leftovers")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    final_dataset = test_dir / "cje_dataset.jsonl"
    if final_dataset.exists():
        dataset_size = count_records(str(final_dataset))
        print(f"✅ Final dataset created with {dataset_size} samples")
    else:
        print("❌ Final dataset not created")

    print("\n✨ Resume functionality test complete!")

    # Clean up
    response = input("\nClean up test data? (y/n): ").strip().lower()
    if response == "y":
        import shutil

        shutil.rmtree(test_dir)
        print("Test data cleaned up.")


if __name__ == "__main__":
    # Make sure we have API keys
    if not os.getenv("FIREWORKS_API_KEY"):
        print("❌ FIREWORKS_API_KEY not set")
        print("Run: source /path/to/set_secrets.sh")
        sys.exit(1)

    main()
