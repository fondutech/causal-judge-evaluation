#!/usr/bin/env python3
"""
Production pipeline for preparing Arena experiment data.

This script runs the complete data preparation pipeline for CJE experiments:
1. Extract prompts from ChatBot Arena dataset
2. Generate responses using different policies
3. Add judge scores (lightweight evaluation)
4. Add oracle labels (high-quality evaluation)
5. Compute log probabilities
6. Prepare CJE dataset with calibration

The prepared data can then be used for ablation studies and analysis.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def run_command(
    cmd: str, check: bool = True, skip_if_exists: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """Run a shell command and return the result.

    Args:
        cmd: Command to run
        check: Whether to raise error on non-zero exit
        skip_if_exists: Skip command if this file exists
    """
    # Check if we should skip
    if skip_if_exists and skip_if_exists.exists():
        print(f"⏭️  Skipping (output exists): {skip_if_exists}")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    # If command starts with "python ", prepend "poetry run "
    if cmd.strip().startswith("python "):
        cmd = "poetry run " + cmd

    print(f"\n📍 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout.strip():
        # Show first 500 chars of output
        output = result.stdout[:500]
        if len(result.stdout) > 500:
            output += "..."
        print(f"Output: {output}")

    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")

    return result


def create_prompts(
    output_dir: Path, n_samples: int, seed: int, skip_existing: bool
) -> None:
    """Extract prompts from ChatBot Arena dataset.

    Args:
        output_dir: Directory to save prompts
        n_samples: Number of prompts to extract
        seed: Random seed for reproducibility
        skip_existing: Whether to skip if prompts file exists
    """
    prompts_file = output_dir / "prompts.jsonl"

    if skip_existing and prompts_file.exists():
        print(f"⏭️  Skipping prompt extraction (file exists): {prompts_file}")
        return

    print(f"Preparing {n_samples} prompts from ChatBot Arena dataset...")

    # Import the prepare function
    from pipeline_steps.prepare_arena_data import prepare_arena_prompts

    # Generate prompts
    prompts = prepare_arena_prompts(
        n_samples=n_samples,
        output_file=str(prompts_file),
        seed=seed,
    )

    if len(prompts) < n_samples:
        print(f"⚠️  Warning: Only got {len(prompts)} prompts, expected {n_samples}")

    print(f"✅ Extracted {len(prompts)} Arena prompts: {prompts_file}")


def main() -> None:
    """Run the production data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare Arena experiment data for CJE analysis"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for output files (default: data)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per response (default: 256)",
    )
    parser.add_argument(
        "--oracle-coverage",
        type=float,
        default=0.5,
        help="Fraction of samples to use for oracle calibration (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps where output files already exist",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the final CJE analysis step",
    )
    args = parser.parse_args()

    print("🚀 Starting Arena experiment data preparation...")
    print(f"   Samples: {args.n_samples}")
    print(f"   Max tokens: {args.max_tokens}")
    print(f"   Oracle coverage: {args.oracle_coverage:.0%}")
    print(f"   Output directory: {args.data_dir}")

    # Setup data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (data_dir / "responses").mkdir(exist_ok=True)
    (data_dir / "logprobs").mkdir(exist_ok=True)

    # Step 1: Extract prompts from Arena dataset
    print("\n" + "=" * 60)
    print("Step 1: Extract prompts from ChatBot Arena")
    print("=" * 60)
    create_prompts(
        data_dir,
        n_samples=args.n_samples,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )

    # Step 2: Generate responses
    print("\n" + "=" * 60)
    print("Step 2: Generate responses for each policy")
    print("=" * 60)

    # Check if all response files exist
    policies = ["base", "clone", "unhelpful"]
    all_responses_exist = all(
        (data_dir / "responses" / f"{policy}_responses.jsonl").exists()
        for policy in policies
    )

    if args.skip_existing and all_responses_exist:
        print("⏭️  Skipping response generation (all files exist)")
    else:
        run_command(
            f"python pipeline_steps/generate_responses.py "
            f"--prompts {data_dir}/prompts.jsonl "
            f"--output-dir {data_dir}/responses "
            f"--max-responses {args.n_samples} "
            f"--max-tokens {args.max_tokens}"
        )

    # Step 3: Add judge scores
    print("\n" + "=" * 60)
    print("Step 3: Add judge scores (lightweight evaluation)")
    print("=" * 60)

    for policy in policies:
        response_file = data_dir / "responses" / f"{policy}_responses.jsonl"

        # Check if judge scores already exist
        if args.skip_existing and response_file.exists():
            with open(response_file) as f:
                first_line = json.loads(f.readline())
                if "judge_score" in first_line.get("metadata", {}):
                    print(f"⏭️  Skipping judge scoring for {policy} (scores exist)")
                    continue

        run_command(
            f"python pipeline_steps/add_judge_scores.py " f"--input {response_file}"
        )

    # Step 4: Add oracle labels
    print("\n" + "=" * 60)
    print("Step 4: Add oracle labels (high-quality evaluation)")
    print("=" * 60)

    for policy in policies:
        response_file = data_dir / "responses" / f"{policy}_responses.jsonl"

        # Check if oracle labels already exist
        if args.skip_existing and response_file.exists():
            with open(response_file) as f:
                first_line = json.loads(f.readline())
                if "oracle_label" in first_line.get("metadata", {}):
                    print(f"⏭️  Skipping oracle labeling for {policy} (labels exist)")
                    continue

        run_command(
            f"python pipeline_steps/add_oracle_labels.py " f"--input {response_file}"
        )

    # Step 5: Compute log probabilities
    print("\n" + "=" * 60)
    print("Step 5: Compute log probabilities")
    print("=" * 60)

    # Check if all logprob files exist
    all_logprobs_exist = all(
        (data_dir / "logprobs" / f"{policy}_logprobs.jsonl").exists()
        for policy in policies
    )

    if args.skip_existing and all_logprobs_exist:
        print("⏭️  Skipping logprob computation (all files exist)")
    else:
        run_command(
            f"python pipeline_steps/compute_logprobs.py "
            f"--responses-dir {data_dir}/responses "
            f"--output-dir {data_dir}/logprobs"
        )

    # Step 6: Prepare CJE dataset
    print("\n" + "=" * 60)
    print("Step 6: Prepare CJE dataset with calibration")
    print("=" * 60)

    cje_dataset_file = data_dir / "cje_dataset.jsonl"

    run_command(
        f"python pipeline_steps/prepare_cje_data.py "
        f"--responses-dir {data_dir}/responses "
        f"--logprobs-dir {data_dir}/logprobs "
        f"--output {cje_dataset_file} "
        f"--oracle-coverage {args.oracle_coverage}",
        skip_if_exists=cje_dataset_file if args.skip_existing else None,
    )

    # Step 7: Run CJE analysis (optional)
    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print("Step 7: Run CJE analysis")
        print("=" * 60)

        results_file = data_dir / "cje_results.json"

        run_command(
            f"python analyze_dataset.py "
            f"--data {cje_dataset_file} "
            f"--n-folds 5 "
            f"--output {results_file}",
            skip_if_exists=results_file if args.skip_existing else None,
        )

        # Print summary if analysis was run
        if results_file.exists():
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

    print("\n✅ Data preparation completed successfully!")
    print(f"📁 Output directory: {data_dir}")
    print("\nNext steps:")
    print(
        f"  1. Run ablation studies: python analyze_oracle_coverage.py --data-dir {data_dir}"
    )
    print(
        f"  2. Analyze specific datasets: python analyze_dataset.py --data {cje_dataset_file}"
    )


if __name__ == "__main__":
    # Check for API keys
    if not os.getenv("FIREWORKS_API_KEY"):
        print("❌ Error: FIREWORKS_API_KEY environment variable not set")
        print("Please run: source /path/to/set_secrets.sh")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "⚠️  Warning: OPENAI_API_KEY not set (required for judge/oracle evaluation)"
        )
        print("Continuing anyway - evaluation steps will fail")

    main()
