#!/usr/bin/env python3
"""
Step 4: Generate responses from target policies.

This consolidated script generates responses for all target policies using the
two-pass teacher forcing approach required for consistent log probabilities.

Features:
- Atomic checkpointing (no duplicates)
- Per-policy, per-sample progress tracking
- Automatic retry on API timeouts
- Configurable batch sizes
- Resume capability

Usage:
    # Generate all policies
    python 04_generate_target_policies.py --input ../data/p0_scored.jsonl --output ../data/target_policies.jsonl

    # Generate specific policy
    python 04_generate_target_policies.py --input ../data/p0_scored.jsonl --output ../data/target_policies.jsonl --policy pi_hot

    # Resume from checkpoint
    python 04_generate_target_policies.py --input ../data/p0_scored.jsonl --output ../data/target_policies.jsonl --resume
"""

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional, Tuple, cast
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.utils.progress import console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)


class CheckpointManager:
    """Manages atomic checkpointing with no duplicates."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_data: Dict[str, Any] = {
            "completed_policies": [],
            "policy_progress": {},
            "completed_samples": {},
            "version": "2.0",
            "last_updated": None,
        }

    def load(self) -> bool:
        """Load checkpoint if it exists. Returns True if loaded."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    self.checkpoint_data = json.load(f)
                return True
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to load checkpoint: {e}[/yellow]")
        return False

    def save(self) -> None:
        """Save checkpoint atomically."""
        self.checkpoint_data["last_updated"] = datetime.now().isoformat()

        # Write to temp file first
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self.checkpoint_data, f, indent=2)

        # Atomic rename
        shutil.move(str(temp_path), str(self.checkpoint_path))

    def is_sample_completed(self, policy: str, sample_idx: int) -> bool:
        """Check if a specific sample for a policy is completed."""
        return sample_idx in self.checkpoint_data.get("completed_samples", {}).get(
            policy, set()
        )

    def mark_sample_completed(self, policy: str, sample_idx: int) -> None:
        """Mark a sample as completed for a policy."""
        if "completed_samples" not in self.checkpoint_data:
            self.checkpoint_data["completed_samples"] = {}
        if policy not in self.checkpoint_data["completed_samples"]:
            self.checkpoint_data["completed_samples"][policy] = []

        completed = self.checkpoint_data["completed_samples"][policy]
        if sample_idx not in completed:
            completed.append(sample_idx)

    def mark_policy_completed(self, policy: str) -> None:
        """Mark entire policy as completed."""
        if policy not in self.checkpoint_data["completed_policies"]:
            self.checkpoint_data["completed_policies"].append(policy)

    def is_policy_completed(self, policy: str) -> bool:
        """Check if entire policy is completed."""
        return policy in self.checkpoint_data.get("completed_policies", [])

    def get_progress(self, policy: str) -> Dict[str, Any]:
        """Get progress info for a policy."""
        progress = self.checkpoint_data.get("policy_progress", {}).get(policy, {})
        return cast(Dict[str, Any], progress)

    def update_progress(self, policy: str, info: Dict[str, Any]) -> None:
        """Update progress info for a policy."""
        if "policy_progress" not in self.checkpoint_data:
            self.checkpoint_data["policy_progress"] = {}
        self.checkpoint_data["policy_progress"][policy] = info


def generate_with_retry(
    runner: APIPolicyRunner,
    prompts: List[str],
    temperature: float,
    max_retries: int = 3,
    use_two_pass: bool = True,
) -> List[Tuple[str, float]]:
    """Generate responses with automatic retry on failure."""

    for attempt in range(max_retries):
        try:
            if use_two_pass:
                # Use two-pass teacher forcing for consistent logprobs
                results = runner.generate_with_consistent_logp(
                    prompts,
                    temperature=temperature,
                    max_new_tokens=1024,
                )
            else:
                # Single pass (only for debugging/comparison)
                results = runner.generate_with_logp(
                    prompts,
                    temperature=temperature,
                    max_new_tokens=1024,
                    return_token_logprobs=False,
                )
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                console.print(
                    f"[yellow]⚠️  API error: {e}. Retrying in {wait_time}s...[/yellow]"
                )
                time.sleep(wait_time)
            else:
                raise
    # This should never be reached due to the raise above
    return []


def generate_policy_responses(
    data: List[Dict[str, Any]],
    policy_name: str,
    policy_config: Dict[str, Any],
    checkpoint_mgr: CheckpointManager,
    batch_size: int = 4,
    use_two_pass: bool = True,
) -> Dict[str, List[Any]]:
    """Generate responses for a single target policy with checkpointing."""

    console.print(f"\n🎯 [bold blue]Generating {policy_name}[/bold blue]")
    console.print(f"   Model: {policy_config['model']}")
    console.print(f"   Temperature: {policy_config['temperature']}")
    console.print(
        f"   Method: {'Two-pass (teacher forcing)' if use_two_pass else 'Single-pass'}"
    )
    if "system_prompt" in policy_config:
        console.print(f"   System: {policy_config['system_prompt'][:50]}...")

    # Initialize runner
    # Determine template format based on model name
    template_format = (
        "llama4" if "llama4" in policy_config["model"].lower() else "llama3"
    )

    runner = APIPolicyRunner(
        provider="fireworks",
        model_name=policy_config["model"],
        temperature=policy_config["temperature"],
        max_new_tokens=1024,
        batch_size=batch_size,
        system_prompt=policy_config.get("system_prompt"),
        completions_template_format=template_format,
    )

    # Validate teacher forcing setup
    console.print(f"   Template: {template_format}")
    try:
        runner.validate_teacher_forcing()
        console.print("   ✅ Teacher forcing validation passed")
    except Exception as e:
        console.print(f"   [red]❌ Teacher forcing validation failed: {e}[/red]")
        console.print("   [yellow]⚠️  Results may be incorrect[/yellow]")

    # Prepare results storage
    responses = []
    logprobs = []

    # Check what we've already completed
    completed_count = 0
    for i in range(len(data)):
        if checkpoint_mgr.is_sample_completed(policy_name, i):
            completed_count += 1
            # Load from data if available
            if f"{policy_name}_response" in data[i]:
                responses.append(data[i][f"{policy_name}_response"])
                logprobs.append(data[i][f"{policy_name}_logprob"])
            else:
                responses.append("")
                logprobs.append(-float("inf"))
        else:
            responses.append(None)
            logprobs.append(None)

    if completed_count == len(data):
        console.print(f"[green]✅ {policy_name} already completed[/green]")
        return {"responses": responses, "logprobs": logprobs}

    console.print(f"   Resuming from: {completed_count}/{len(data)} samples")

    # Generate missing responses
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(f"Generating {policy_name}", total=len(data))
        progress.update(task, completed=completed_count)

        # Process in batches
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch_indices = list(range(batch_start, batch_end))

            # Filter to only non-completed samples
            indices_to_process = [
                i
                for i in batch_indices
                if not checkpoint_mgr.is_sample_completed(policy_name, i)
            ]

            if not indices_to_process:
                continue

            # Extract prompts for this batch
            batch_prompts = [data[i]["prompt"] for i in indices_to_process]

            try:
                # Generate with retry
                batch_results = generate_with_retry(
                    runner,
                    batch_prompts,
                    policy_config["temperature"],
                    use_two_pass=use_two_pass,
                )

                # Store results
                for idx, (response, logprob) in zip(indices_to_process, batch_results):
                    responses[idx] = response
                    logprobs[idx] = logprob
                    checkpoint_mgr.mark_sample_completed(policy_name, idx)
                    progress.update(task, advance=1)

                # Save checkpoint after each batch
                checkpoint_mgr.save()

            except Exception as e:
                console.print(
                    f"\n[red]❌ Failed to generate batch {batch_start}-{batch_end}: {e}[/red]"
                )
                # Continue with next batch instead of failing completely
                continue

            # Rate limit protection
            if batch_end < len(data):
                time.sleep(0.5)

    # Verify all completed
    if all(r is not None for r in responses):
        checkpoint_mgr.mark_policy_completed(policy_name)
        checkpoint_mgr.save()
        console.print(f"[green]✅ {policy_name} generation complete[/green]")
    else:
        incomplete = sum(1 for r in responses if r is None)
        console.print(
            f"[yellow]⚠️  {policy_name} incomplete: {incomplete} samples failed[/yellow]"
        )

    return {"responses": responses, "logprobs": logprobs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate target policy responses")
    parser.add_argument(
        "--input", default="../data/p0_scored.jsonl", help="Input file with prompts"
    )
    parser.add_argument(
        "--output", default="../data/target_policies.jsonl", help="Output file"
    )
    parser.add_argument("--policy", type=str, help="Generate only this specific policy")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for API calls"
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Use single-pass generation (NOT RECOMMENDED)",
    )
    args = parser.parse_args()

    console.print("[bold blue]🚀 Step 4: Generate Target Policy Responses[/bold blue]")

    if args.single_pass:
        console.print(
            "[bold red]⚠️  WARNING: Using single-pass generation. This violates causal identification requirements![/bold red]"
        )
        console.print("[red]Only use this for debugging. Results will be biased.[/red]")

    # Load input data
    data = []
    with open(args.input, "r") as f:
        for line in f:
            data.append(json.loads(line))

    console.print(f"📄 Loaded {len(data)} samples from {args.input}")

    # Initialize checkpoint manager
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.json")
    checkpoint_mgr = CheckpointManager(checkpoint_path)

    if args.resume or checkpoint_mgr.load():
        console.print(f"📥 Loaded checkpoint from {checkpoint_path}")
        completed = checkpoint_mgr.checkpoint_data.get("completed_policies", [])
        if completed:
            console.print(f"   Completed policies: {', '.join(completed)}")

    # Define target policies (from arena_10k.yaml)
    target_policies = {
        "pi_clone": {
            "model": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "temperature": 0.5,
            "description": "Clone of π₀ (sanity check)",
        },
        "pi_cot": {
            "model": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "temperature": 0.5,
            "system_prompt": "You are a helpful assistant.\n\nThink step-by-step before providing your answer.",
            "description": "Chain-of-thought prompting",
        },
        "pi_bigger_model": {
            "model": "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "temperature": 0.5,
            "description": "Larger model (Maverick)",
        },
    }

    # Filter to single policy if requested
    if args.policy:
        if args.policy not in target_policies:
            console.print(f"[red]❌ Unknown policy: {args.policy}[/red]")
            console.print(f"Available: {', '.join(target_policies.keys())}")
            return 1
        target_policies = {args.policy: target_policies[args.policy]}

    # Generate for each policy
    start_time = time.time()

    for policy_name, policy_config in target_policies.items():
        if checkpoint_mgr.is_policy_completed(policy_name):
            console.print(f"\n⏭️  Skipping {policy_name} (already completed)")
            continue

        results = generate_policy_responses(
            data,
            policy_name,
            policy_config,
            checkpoint_mgr,
            batch_size=args.batch_size,
            use_two_pass=not args.single_pass,
        )

        # Add results to data
        for i, (response, logprob) in enumerate(
            zip(results["responses"], results["logprobs"])
        ):
            if response is not None:
                data[i][f"{policy_name}_response"] = response
                data[i][f"{policy_name}_logprob"] = logprob

    # Save final output
    console.print(f"\n💾 Saving to {args.output}")
    with open(args.output, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Summary
    elapsed = time.time() - start_time
    console.print(f"\n⏱️  Total time: {elapsed/60:.1f} minutes")

    console.print("\n📊 [bold]Summary[/bold]")
    console.print(f"Total samples: {len(data)}")

    # Check what policies we have
    if data:
        sample = data[0]
        for policy in ["pi_clone", "pi_cot", "pi_bigger_model"]:
            if f"{policy}_response" in sample:
                valid_responses = sum(
                    1
                    for d in data
                    if f"{policy}_response" in d and d[f"{policy}_response"]
                )
                console.print(f"✓ {policy}: {valid_responses}/{len(data)} responses")
            else:
                console.print(f"✗ {policy}: not generated")

    # Clean up checkpoint if everything completed
    all_complete = all(
        checkpoint_mgr.is_policy_completed(p)
        for p in ["pi_clone", "pi_cot", "pi_bigger_model"]
        if args.policy is None or p == args.policy
    )

    if all_complete and checkpoint_path.exists():
        checkpoint_path.unlink()
        console.print("\n🧹 Cleaned up checkpoint file")

    console.print("\n✅ Target policy generation complete!")
    console.print("Next: Export for oracle labeling (step 5)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
