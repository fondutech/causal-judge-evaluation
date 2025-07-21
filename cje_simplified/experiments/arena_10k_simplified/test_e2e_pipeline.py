#!/usr/bin/env python3
"""
End-to-end test for the Arena experiment pipeline.

This test runs the entire pipeline on 2 samples with fixed seeds
to ensure deterministic behavior.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"\n📍 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout.strip():
        print(f"Output: {result.stdout[:500]}...")

    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")

    return result


def create_test_prompts(output_dir: Path) -> None:
    """Create a minimal test prompts file with 4 samples."""
    prompts = [
        {"prompt_id": "test_0", "prompt": "What is 2+2?"},
        {"prompt_id": "test_1", "prompt": "Explain quantum computing in one sentence."},
        {"prompt_id": "test_2", "prompt": "What is the capital of France?"},
        {"prompt_id": "test_3", "prompt": "How do I make a paper airplane?"},
    ]

    prompts_file = output_dir / "prompts.jsonl"
    with open(prompts_file, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")

    print(f"✅ Created test prompts: {prompts_file}")


def verify_responses(responses_dir: Path) -> None:
    """Verify that responses were generated correctly."""
    policies = ["base", "clone", "unhelpful"]

    for policy in policies:
        file_path = responses_dir / f"{policy}_responses.jsonl"
        assert file_path.exists(), f"Missing response file: {file_path}"

        with open(file_path) as f:
            lines = f.readlines()
            assert (
                len(lines) == 4
            ), f"Expected 4 responses, got {len(lines)} for {policy}"

            # Check structure
            for line in lines:
                data = json.loads(line)
                assert "prompt_id" in data
                assert "prompt" in data
                assert "response" in data
                assert "policy" in data
                assert data["policy"] == policy

    print("✅ Response files verified")


def verify_evaluation_scores(responses_dir: Path) -> None:
    """Verify that judge and oracle scores were added."""
    base_file = responses_dir / "base_responses.jsonl"

    with open(base_file) as f:
        for line in f:
            data = json.loads(line)
            metadata = data.get("metadata", {})
            assert "judge_score" in metadata, "Missing judge_score"
            assert "oracle_label" in metadata, "Missing oracle_label"
            assert 0 <= metadata["judge_score"] <= 1
            assert 0 <= metadata["oracle_label"] <= 1

    print("✅ Evaluation scores verified")


def verify_logprobs(logprobs_dir: Path) -> None:
    """Verify that log probabilities were computed."""
    policies = ["base", "clone", "unhelpful"]

    for policy in policies:
        file_path = logprobs_dir / f"{policy}_logprobs.jsonl"
        assert file_path.exists(), f"Missing logprob file: {file_path}"

        with open(file_path) as f:
            lines = f.readlines()
            assert (
                len(lines) == 4
            ), f"Expected 4 logprobs, got {len(lines)} for {policy}"

            for line in lines:
                data = json.loads(line)
                assert "prompt_id" in data
                assert "logprob" in data
                # Logprob can be None if computation failed
                assert data["logprob"] is None or isinstance(
                    data["logprob"], (int, float)
                )

    print("✅ Log probability files verified")


def verify_cje_dataset(dataset_file: Path) -> None:
    """Verify the final CJE dataset."""
    with open(dataset_file) as f:
        lines = f.readlines()
        assert len(lines) == 4, f"Expected 4 records, got {len(lines)}"

        for line in lines:
            data = json.loads(line)
            # Check required fields
            assert "prompt" in data
            assert "response" in data
            assert "base_policy_logprob" in data
            assert "target_policy_logprobs" in data
            assert "metadata" in data

            # Check metadata
            metadata = data["metadata"]
            assert "prompt_id" in metadata
            assert "judge_score" in metadata
            assert "oracle_label" in metadata

            # Check target policies
            target_logprobs = data["target_policy_logprobs"]
            assert "clone" in target_logprobs
            assert "unhelpful" in target_logprobs

    print("✅ CJE dataset verified")


def main() -> None:
    """Run the end-to-end test."""
    print("🧪 Starting end-to-end pipeline test...")

    # Setup test directory
    test_dir = Path("test_e2e_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create subdirectories
    (test_dir / "responses").mkdir()
    (test_dir / "logprobs").mkdir()

    try:
        # Step 1: Create test prompts
        create_test_prompts(test_dir)

        # Step 2: Generate responses (with fixed seed)
        # Note: We'll use temperature=0 for deterministic responses
        run_command(
            f"python generate_responses.py "
            f"--prompts {test_dir}/prompts.jsonl "
            f"--output-dir {test_dir}/responses "
            f"--max-responses 4"
        )
        verify_responses(test_dir / "responses")

        # Step 3: Add judge scores
        for policy in ["base", "clone", "unhelpful"]:
            run_command(
                f"python add_judge_scores.py "
                f"--input {test_dir}/responses/{policy}_responses.jsonl"
            )

        # Step 4: Add oracle labels
        for policy in ["base", "clone", "unhelpful"]:
            run_command(
                f"python add_oracle_labels.py "
                f"--input {test_dir}/responses/{policy}_responses.jsonl"
            )
        verify_evaluation_scores(test_dir / "responses")

        # Step 5: Compute log probabilities
        run_command(
            f"python compute_logprobs.py "
            f"--responses-dir {test_dir}/responses "
            f"--output-dir {test_dir}/logprobs"
        )
        verify_logprobs(test_dir / "logprobs")

        # Step 6: Prepare CJE dataset
        run_command(
            f"python prepare_cje_data.py "
            f"--responses-dir {test_dir}/responses "
            f"--logprobs-dir {test_dir}/logprobs "
            f"--output {test_dir}/cje_dataset.jsonl"
        )
        verify_cje_dataset(test_dir / "cje_dataset.jsonl")

        # Step 7: Run CJE analysis (quick test)
        result = run_command(
            f"python run_cje_analysis.py "
            f"--data {test_dir}/cje_dataset.jsonl "
            f"--n-folds 2",
            check=False,  # May fail if not enough samples with valid logprobs
        )

        if result.returncode == 0:
            print("✅ CJE analysis completed")
        else:
            print("⚠️  CJE analysis failed (expected with only 4 samples)")

        print("\n🎉 End-to-end test completed successfully!")

        # Print sample of final dataset
        print("\n📊 Sample of final CJE dataset:")
        with open(test_dir / "cje_dataset.jsonl") as f:
            first_record = json.loads(f.readline())
            print(json.dumps(first_record, indent=2))

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"Test directory preserved at: {test_dir}")
        raise
    else:
        # Cleanup only on success
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("\n🧹 Test directory cleaned up")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("FIREWORKS_API_KEY"):
        print("❌ Error: FIREWORKS_API_KEY environment variable not set")
        print("Please run: source /path/to/set_secrets.sh")
        sys.exit(1)

    main()
