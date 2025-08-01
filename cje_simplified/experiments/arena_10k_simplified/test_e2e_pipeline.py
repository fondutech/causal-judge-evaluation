#!/usr/bin/env python3
"""
End-to-end test for the Arena experiment pipeline.

This test runs the entire pipeline on 10 samples with fixed seeds
to ensure deterministic behavior.
"""

import argparse
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
    # If command starts with "python ", prepend "poetry run "
    if cmd.strip().startswith("python "):
        cmd = "poetry run " + cmd

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


def create_test_prompts(output_dir: Path, n_samples: int = 50) -> None:
    """Extract test prompts from ChatBot Arena dataset.

    Args:
        output_dir: Directory to save prompts
        n_samples: Number of prompts to extract (default: 50)
    """
    print(f"Preparing {n_samples} prompts from ChatBot Arena dataset...")

    # Import the prepare function
    from pipeline_steps.prepare_arena_data import prepare_arena_prompts

    # Generate prompts with a test-specific seed for reproducibility
    prompts_file = output_dir / "prompts.jsonl"
    prompts = prepare_arena_prompts(
        n_samples=n_samples,
        output_file=str(prompts_file),
        seed=12345,  # Different seed for test data
    )

    if len(prompts) < n_samples:
        print(f"⚠️  Warning: Only got {len(prompts)} prompts, expected {n_samples}")

    print(f"✅ Extracted {len(prompts)} Arena prompts: {prompts_file}")


def verify_responses(responses_dir: Path, n_samples: int = 50) -> None:
    """Verify that responses were generated correctly."""
    policies = ["base", "clone", "unhelpful"]

    for policy in policies:
        file_path = responses_dir / f"{policy}_responses.jsonl"
        assert file_path.exists(), f"Missing response file: {file_path}"

        with open(file_path) as f:
            lines = f.readlines()
            assert (
                len(lines) == n_samples
            ), f"Expected {n_samples} responses, got {len(lines)} for {policy}"

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


def verify_logprobs(logprobs_dir: Path, n_samples: int = 50) -> None:
    """Verify that log probabilities were computed."""
    policies = ["base", "clone", "unhelpful"]

    for policy in policies:
        file_path = logprobs_dir / f"{policy}_logprobs.jsonl"
        assert file_path.exists(), f"Missing logprob file: {file_path}"

        with open(file_path) as f:
            lines = f.readlines()
            assert (
                len(lines) == n_samples
            ), f"Expected {n_samples} logprobs, got {len(lines)} for {policy}"

            for line in lines:
                data = json.loads(line)
                assert "prompt_id" in data
                assert "logprob" in data
                # Logprob can be None if computation failed
                assert data["logprob"] is None or isinstance(
                    data["logprob"], (int, float)
                )

    print("✅ Log probability files verified")


def verify_cje_dataset(dataset_file: Path, n_samples: int = 50) -> None:
    """Verify the final CJE dataset."""
    with open(dataset_file) as f:
        lines = f.readlines()
        assert (
            len(lines) == n_samples
        ), f"Expected {n_samples} records, got {len(lines)}"

        for line in lines:
            data = json.loads(line)
            # Check required fields
            assert "prompt" in data
            assert "response" in data
            assert "base_policy_logprob" in data
            assert "target_policy_logprobs" in data
            assert "metadata" in data
            assert "reward" in data, "Missing reward field"

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
    parser = argparse.ArgumentParser(description="End-to-end test for CJE pipeline")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't clean up test directory after completion",
    )
    parser.add_argument(
        "--test-dir",
        default="test_e2e_data",
        help="Directory for test files (default: test_e2e_data)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to test (default: 50)",
    )
    args = parser.parse_args()

    print("🧪 Starting end-to-end pipeline test...")

    # Setup test directory
    test_dir = Path(args.test_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create subdirectories
    (test_dir / "responses").mkdir()
    (test_dir / "logprobs").mkdir()

    try:
        # Step 1: Extract test prompts from Arena dataset
        # Using real Arena prompts ensures realistic diversity and complexity
        create_test_prompts(test_dir, n_samples=args.n_samples)

        # Step 2: Generate responses (with fixed seed)
        # Note: Using max-tokens=50 for faster testing (enough for 1-2 sentences)
        run_command(
            f"python pipeline_steps/generate_responses.py "
            f"--prompts {test_dir}/prompts.jsonl "
            f"--output-dir {test_dir}/responses "
            f"--max-responses {args.n_samples} "
            f"--max-tokens 50"
        )
        verify_responses(test_dir / "responses", n_samples=args.n_samples)

        # Step 3: Add judge scores
        for policy in ["base", "clone", "unhelpful"]:
            run_command(
                f"python pipeline_steps/add_judge_scores.py "
                f"--input {test_dir}/responses/{policy}_responses.jsonl"
            )

        # Step 4: Add oracle labels
        for policy in ["base", "clone", "unhelpful"]:
            run_command(
                f"python pipeline_steps/add_oracle_labels.py "
                f"--input {test_dir}/responses/{policy}_responses.jsonl"
            )
        verify_evaluation_scores(test_dir / "responses")

        # Step 5: Compute log probabilities
        run_command(
            f"python pipeline_steps/compute_logprobs.py "
            f"--responses-dir {test_dir}/responses "
            f"--output-dir {test_dir}/logprobs"
        )
        verify_logprobs(test_dir / "logprobs", n_samples=args.n_samples)

        # Step 6: Prepare CJE dataset
        # Note: Using 50% oracle coverage to test calibration workflow
        run_command(
            f"python pipeline_steps/prepare_cje_data.py "
            f"--responses-dir {test_dir}/responses "
            f"--logprobs-dir {test_dir}/logprobs "
            f"--output {test_dir}/cje_dataset.jsonl "
            f"--oracle-coverage 0.5"
        )
        verify_cje_dataset(test_dir / "cje_dataset.jsonl", n_samples=args.n_samples)

        # Step 7: Run CJE analysis (quick test)
        results_file = test_dir / "cje_results.json"
        result = run_command(
            f"python run_cje_analysis.py "
            f"--data {test_dir}/cje_dataset.jsonl "
            f"--n-folds 2 "
            f"--output {results_file}",
            check=True,  # Should succeed with 10 samples
        )

        print("✅ CJE analysis completed")

        # Verify results file was created
        assert results_file.exists(), "CJE results file not created"
        print(f"✓ CJE results written to: {results_file}")

        # Print summary of results
        with open(results_file) as f:
            results_data = json.load(f)
            print("\n📊 CJE Results Summary:")
            print(f"  Best policy: {results_data['best_policy']}")
            print(
                f"  Effective sample size: {results_data['weight_diagnostics']['effective_sample_size']:.0f}"
            )
            for policy, stats in results_data["estimation"]["policies"].items():
                print(
                    f"  {policy}: {stats['estimate']:.3f} ± {stats['standard_error']:.3f}"
                )

        print("\n🎉 End-to-end test completed successfully!")

        # Print sample of final dataset
        print("\n📊 Sample of final CJE dataset:")
        with open(test_dir / "cje_dataset.jsonl") as f:
            first_record = json.loads(f.readline())
            print(json.dumps(first_record, indent=2))

        if not args.no_cleanup:
            print(
                "\n💡 Tip: Use --no-cleanup flag to preserve test files for inspection"
            )

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"Test directory preserved at: {test_dir}")
        raise
    else:
        # Cleanup only on success and if not disabled
        if args.no_cleanup:
            print(f"\n📁 Test directory preserved at: {test_dir}")
            print("  Run without --no-cleanup flag to automatically clean up")
        else:
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
