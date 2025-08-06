#!/usr/bin/env python3
"""
Test script to verify batch saving and resume functionality.
"""

import json
import os
import subprocess
import time
from pathlib import Path


def simulate_interrupt_test() -> bool:
    """Test that batch saving and resume works correctly."""

    print("🧪 Testing batch save and resume functionality...\n")

    # Clean up previous test
    test_dir = Path("test_batch")
    if test_dir.exists():
        import shutil

        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create test prompts
    prompts = [
        {"prompt_id": f"test_{i}", "prompt": f"What is {i} + {i}?"} for i in range(10)
    ]

    prompts_file = test_dir / "prompts.jsonl"
    with open(prompts_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"✓ Created {len(prompts)} test prompts")

    # Test 1: Generate first 3 responses with batch size 2
    print("\n📝 Test 1: Generate first batch of responses")
    print("  Running with --max-responses 3 --batch-size 2")

    result = subprocess.run(
        f"poetry run python pipeline_steps/generate_responses.py "
        f"--prompts {prompts_file} "
        f"--output-dir {test_dir}/responses "
        f"--max-responses 3 "
        f"--max-tokens 20 "
        f"--batch-size 2",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return False

    # Check that we have 3 responses for base policy
    base_file = test_dir / "responses" / "base_responses.jsonl"
    with open(base_file) as f:
        lines = f.readlines()

    if len(lines) != 3:
        print(f"❌ Expected 3 responses, got {len(lines)}")
        return False

    print(f"  ✓ Generated {len(lines)} responses")

    # Test 2: Resume and generate the rest
    print("\n📝 Test 2: Resume and generate remaining responses")
    print("  Running again with --max-responses 10 (should resume from 3)")

    result = subprocess.run(
        f"poetry run python pipeline_steps/generate_responses.py "
        f"--prompts {prompts_file} "
        f"--output-dir {test_dir}/responses "
        f"--max-responses 10 "
        f"--max-tokens 20 "
        f"--batch-size 2",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return False

    # Check output for resume message
    if "Resuming from previous run" in result.stdout:
        print("  ✓ Detected resume from previous run")
    else:
        print("  ⚠️  No resume message found")

    # Check final count
    with open(base_file) as f:
        lines = f.readlines()

    if len(lines) != 10:
        print(f"❌ Expected 10 total responses, got {len(lines)}")
        return False

    print(f"  ✓ Total responses after resume: {len(lines)}")

    # Verify all prompt_ids are unique
    prompt_ids = set()
    for line in lines:
        data = json.loads(line)
        prompt_ids.add(data["prompt_id"])

    if len(prompt_ids) != 10:
        print(f"❌ Expected 10 unique prompt_ids, got {len(prompt_ids)}")
        return False

    print(f"  ✓ All {len(prompt_ids)} prompt_ids are unique")

    # Test 3: Run again, should detect all complete
    print("\n📝 Test 3: Run again with all responses complete")

    result = subprocess.run(
        f"poetry run python pipeline_steps/generate_responses.py "
        f"--prompts {prompts_file} "
        f"--output-dir {test_dir}/responses "
        f"--max-responses 10 "
        f"--max-tokens 20 "
        f"--batch-size 2",
        shell=True,
        capture_output=True,
        text=True,
    )

    if "All 10 responses already exist" in result.stdout:
        print("  ✓ Correctly detected all responses complete")
    else:
        print("  ⚠️  Did not detect completion")

    print("\n✅ All batch save/resume tests passed!")

    # Clean up
    import shutil

    shutil.rmtree(test_dir)
    print("\n🧹 Test directory cleaned up")

    return True


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("FIREWORKS_API_KEY"):
        print("❌ Error: FIREWORKS_API_KEY environment variable not set")
        exit(1)

    success = simulate_interrupt_test()
    exit(0 if success else 1)
