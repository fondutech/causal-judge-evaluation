#!/usr/bin/env python3
"""
Step 2b: Compute log probabilities for P0 responses under all policies.

This script computes log P(P0_response|prompt, policy) for each policy
using robust teacher forcing. Critical for importance weighting.

Includes extreme weight detection and automatic rejection to prevent
corruption from token boundary bugs.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils import RobustTeacherForcing, CheckpointManager, BatchProcessor
from cje.utils.progress import console
from cje.types.results import LogProbResult, LogProbStatus
from config_loader import load_arena_config


# Validation configuration
MAX_ABS_LOG_RATIO = 2.0  # ~7x weight - anything beyond this is suspicious
EXTREME_LOG_RATIO = 5.0  # ~150x weight - definitely wrong
CHECK_POLICIES = {"pi_clone"}  # Pi_clone should have weight ~1.0


def validate_log_probabilities(
    prompt_id: str,
    prompt: str,
    response: str,
    logprobs: Dict[str, Optional[float]],
    reject_extreme: bool = True,
    log_file: str = "data/extreme_weights.jsonl",
) -> Tuple[Dict[str, Optional[float]], bool, List[Dict[str, Any]]]:
    """
    Validate computed log probabilities and flag extreme weights.

    Args:
        prompt_id: Identifier for the prompt
        prompt: The prompt text
        response: The response text
        logprobs: Dict of {policy: log_prob}
        reject_extreme: If True, set extreme values to None
        log_file: File to log extreme cases

    Returns:
        validated_logprobs: Dict with extreme values possibly set to None
        has_issues: Boolean indicating if issues were found
        issues: List of issue dictionaries
    """
    validated = logprobs.copy()
    issues = []

    # Get P0 baseline
    p0_logp = logprobs.get("p0")
    if p0_logp is None:
        return validated, True, []

    # Check each policy
    for policy, logp in logprobs.items():
        if logp is None or policy not in CHECK_POLICIES:
            continue

        log_ratio = logp - p0_logp
        weight = 2.718281828**log_ratio

        if abs(log_ratio) > MAX_ABS_LOG_RATIO:
            issue = {
                "prompt_id": prompt_id,
                "policy": policy,
                "p0_logp": p0_logp,
                "policy_logp": logp,
                "log_ratio": log_ratio,
                "weight": weight,
                "prompt_preview": prompt[:100],
                "response_preview": response[:100],
                "response_length": len(response),
            }

            # Determine severity
            if abs(log_ratio) > EXTREME_LOG_RATIO:
                issue["severity"] = "EXTREME"
                console.print(
                    f"[red]❌ EXTREME weight for {policy} on {prompt_id}: "
                    f"{weight:.2e} (log ratio: {log_ratio:.2f})[/red]"
                )

                if reject_extreme:
                    validated[policy] = None
                    issue["action"] = "REJECTED"
            else:
                issue["severity"] = "WARNING"
                console.print(
                    f"[yellow]⚠️  High weight for {policy} on {prompt_id}: "
                    f"{weight:.2f} (log ratio: {log_ratio:.2f})[/yellow]"
                )
                issue["action"] = "KEPT"

            issues.append(issue)

    # Log issues to file
    if issues and log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            for issue in issues:
                f.write(json.dumps(issue) + "\n")

    return validated, len(issues) > 0, issues


def compute_logprobs_batch(
    batch: List[Dict[str, Any]],
    teacher_forcing_instances: Dict[str, RobustTeacherForcing],
    policy_configs: Dict[str, Dict[str, Any]],
    reject_extreme: bool = True,
) -> List[Dict[str, Any]]:
    """Compute log probabilities for a batch of items with validation."""
    results = []
    validation_stats = {"total": 0, "warnings": 0, "extreme": 0, "rejected": 0}

    for item in batch:
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]
        p0_response = item["p0_response"]
        validation_stats["total"] += 1

        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "p0_response": p0_response,
            "logprobs": {},
        }

        # Compute P0 response log prob under each policy
        for policy_name, tf_instance in teacher_forcing_instances.items():
            logprob_result = tf_instance.compute_log_prob(prompt, p0_response)

            if logprob_result.status == LogProbStatus.SUCCESS:
                result["logprobs"][policy_name] = logprob_result.value
            else:
                result["logprobs"][policy_name] = None
                console.print(
                    f"[yellow]Warning: Failed to compute {policy_name} log prob "
                    f"for prompt {prompt_id}: {logprob_result.error}[/yellow]"
                )

        # Validate before saving
        validated_logprobs, has_issues, issues = validate_log_probabilities(
            prompt_id=prompt_id,
            prompt=prompt,
            response=p0_response,
            logprobs=result["logprobs"],
            reject_extreme=reject_extreme,
        )

        result["logprobs"] = validated_logprobs

        # Track validation statistics
        if has_issues:
            for issue in issues:
                if issue["severity"] == "EXTREME":
                    validation_stats["extreme"] += 1
                    if issue.get("action") == "REJECTED":
                        validation_stats["rejected"] += 1
                else:
                    validation_stats["warnings"] += 1

        results.append(result)

    # Print batch validation summary
    if validation_stats["warnings"] > 0 or validation_stats["extreme"] > 0:
        console.print(
            f"\n📊 Batch validation: "
            f"{validation_stats['warnings']} warnings, "
            f"{validation_stats['extreme']} extreme ({validation_stats['rejected']} rejected)"
        )

    return results


def main():
    # No arguments - fixed paths from convention
    INPUT_FILE = "data/all_responses.jsonl"
    OUTPUT_FILE = "data/logprobs.jsonl"

    console.print(
        "[bold cyan]Step 2b: Compute Log Probabilities (with validation)[/bold cyan]"
    )

    # Check API key
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        console.print("❌ [red]Error: FIREWORKS_API_KEY not set![/red]")
        sys.exit(1)

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(
            f"❌ [red]Error: {INPUT_FILE} not found. Run 02_generate_responses.py first.[/red]"
        )
        sys.exit(1)

    # Load config
    config = load_arena_config()

    # Build full policy configurations
    policy_configs = {"p0": config.logging_policy}
    for policy in config.target_policies:
        policy_configs[policy["name"]] = policy

    # Load all responses
    console.print(f"\n📄 Loading all responses from {INPUT_FILE}")
    all_responses_data = []
    with open(INPUT_FILE) as f:
        for line in f:
            all_responses_data.append(json.loads(line))
    console.print(f"✅ Loaded {len(all_responses_data)} prompt entries")

    # Extract P0 responses for processing
    p0_data = []
    for entry in all_responses_data:
        if "p0" in entry["responses"]:
            p0_resp = entry["responses"]["p0"]
            p0_data.append(
                {
                    "prompt_id": entry["prompt_id"],
                    "prompt": entry["prompt"],
                    "p0_response": p0_resp["response"],
                    "metadata": entry.get("metadata", {}),
                }
            )

    console.print(f"📊 Processing {len(p0_data)} P0 responses")

    # Initialize teacher forcing for each policy with full config
    console.print("\n🤖 Initializing teacher forcing for all policies:")
    tf_instances = {}
    for policy_name, policy_config in policy_configs.items():
        # Extract relevant parameters
        model_name = policy_config["model_name"]
        temperature = policy_config.get("temperature", 0.5)
        system_prompt = policy_config.get("system_prompt", None)

        tf_instances[policy_name] = RobustTeacherForcing(
            provider="fireworks",
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            system_prompt=system_prompt,
            force_continuation=True,  # ONLY continuation method - no fallback to token counting
        )
        console.print(f"  ✅ {policy_name}: {model_name} (temp={temperature})")
        if system_prompt:
            console.print(f"     └─ System prompt: {system_prompt[:50]}...")

    # Clear previous extreme weights log
    extreme_log_file = "data/extreme_weights.jsonl"
    if Path(extreme_log_file).exists():
        Path(extreme_log_file).unlink()
        console.print(f"🧹 Cleared previous extreme weights log")

    # Process with checkpointing
    console.print(f"\n🔄 Computing log probabilities with validation...")
    console.print(f"⚠️  Will reject extreme weights (|log ratio| > {EXTREME_LOG_RATIO})")
    console.print(
        f"🛡️  Using ONLY continuation method (no fallback) for maximum reliability"
    )

    checkpoint_mgr = CheckpointManager(
        checkpoint_path="data/checkpoint_logprobs.jsonl",
        get_uid_fn=lambda x: x["prompt_id"],
    )

    processor = BatchProcessor(
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,
    )

    results = processor.process_batches(
        p0_data,
        lambda batch: compute_logprobs_batch(
            batch, tf_instances, policy_configs, reject_extreme=True
        ),
        description="Computing log probs",
    )

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Validate results and show final statistics
    stats_by_policy = defaultdict(
        lambda: {"valid": 0, "null": 0, "zero": 0, "logprobs": []}
    )
    zero_samples = []
    validation_summary = {"total": 0, "warnings": 0, "extreme": 0, "rejected": 0}

    for item in results:
        prompt_id = item["prompt_id"]
        p0_response = item.get("p0_response", "")
        validation_summary["total"] += 1

        for policy, logprob in item["logprobs"].items():
            if logprob is not None:
                stats_by_policy[policy]["valid"] += 1
                stats_by_policy[policy]["logprobs"].append(logprob)
                if logprob == 0.0 and len(p0_response.strip()) > 0:
                    stats_by_policy[policy]["zero"] += 1
                    zero_samples.append((prompt_id, policy, p0_response[:50]))
            else:
                stats_by_policy[policy]["null"] += 1

    # Check extreme weights log
    if Path(extreme_log_file).exists():
        with open(extreme_log_file) as f:
            extreme_issues = [json.loads(line) for line in f]

        for issue in extreme_issues:
            if issue["severity"] == "EXTREME":
                validation_summary["extreme"] += 1
                if issue.get("action") == "REJECTED":
                    validation_summary["rejected"] += 1
            else:
                validation_summary["warnings"] += 1

    # Print statistics
    console.print("\n📊 Log Probability Statistics:")
    for policy in sorted(stats_by_policy.keys()):
        stats = stats_by_policy[policy]
        if stats["logprobs"]:
            import numpy as np

            mean = np.mean(stats["logprobs"])
            std = np.std(stats["logprobs"])
            console.print(
                f"  {policy}: valid={stats['valid']}, failed={stats['null']}, "
                f"zero={stats['zero']}, mean={mean:.2f}, std={std:.2f}"
            )

    # Print validation summary
    console.print("\n🛡️  Validation Summary:")
    console.print(f"  Total samples: {validation_summary['total']}")
    console.print(f"  Warnings (kept): {validation_summary['warnings']}")
    console.print(f"  Extreme (detected): {validation_summary['extreme']}")
    console.print(f"  Extreme (rejected): {validation_summary['rejected']}")

    if validation_summary["extreme"] > 0:
        console.print(
            f"\n[yellow]⚠️  See {extreme_log_file} for details on extreme weights[/yellow]"
        )

    # Warn about zero values
    if zero_samples:
        console.print(
            f"\n[yellow]⚠️  Found {len(zero_samples)} samples with 0.0 log probs "
            f"for non-empty responses[/yellow]"
        )
        console.print("[yellow]First 3 examples:[/yellow]")
        for prompt_id, policy, response_preview in zero_samples[:3]:
            console.print(f"  - {prompt_id} ({policy}): {response_preview}...")

    # Show teacher forcing statistics
    console.print("\n📊 Teacher Forcing Method Usage:")
    for policy_name, tf in tf_instances.items():
        stats = tf.get_stats()
        if stats["total_calls"] > 0:
            console.print(f"  {policy_name}: {stats}")

    console.print(f"\n✅ [bold green]Saved results to {OUTPUT_FILE}[/bold green]")

    # Clean up checkpoint
    checkpoint_path = Path("data/checkpoint_logprobs.jsonl")
    if checkpoint_path.exists():
        console.print(f"🧹 Cleaning up checkpoint file")
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
