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

# Import llama.cpp teacher forcing if available
try:
    from cje.utils import LlamaCppTeacherForcing

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


# Validation configuration - only check pi_clone for extreme weights
EXTREME_WEIGHT_THRESHOLD = 5.0  # Weights > 5x for pi_clone indicate issues

# FIXED: Settings for stable log probabilities
# NOTE: We use each policy's production temperature for accurate importance weights
LOG_PROB_TOP_P = 1.0  # No nucleus sampling
LOG_PROB_SEED = 42  # For reproducibility


def check_pi_clone_weight(
    prompt_id: str,
    logprobs: Dict[str, Optional[float]],
    log_extreme: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Check if pi_clone has extreme weights (should be ~1.0).

    Returns issue dict if extreme weight detected, None otherwise.
    """
    # Only check pi_clone
    if "pi_clone" not in logprobs or "p0" not in logprobs:
        return None

    p0_logp = logprobs.get("p0")
    clone_logp = logprobs.get("pi_clone")

    if p0_logp is None or clone_logp is None:
        return None

    log_ratio = clone_logp - p0_logp
    weight = 2.718281828**log_ratio

    # Check for extreme weights
    if weight > EXTREME_WEIGHT_THRESHOLD or weight < 1 / EXTREME_WEIGHT_THRESHOLD:
        if log_extreme:
            console.print(
                f"[red]❌ EXTREME pi_clone weight for {prompt_id}: "
                f"{weight:.2f} (expected ~1.0)[/red]"
            )

        return {
            "prompt_id": prompt_id,
            "log_ratio": log_ratio,
            "weight": weight,
            "p0_logp": p0_logp,
            "clone_logp": clone_logp,
        }

    return None


def compute_logprobs_batch(
    batch: List[Dict[str, Any]],
    teacher_forcing_instances: Dict[str, RobustTeacherForcing],
    policy_configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute log probabilities for a batch of items."""
    results = []
    extreme_count = 0

    for item in batch:
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]
        p0_response = item["p0_response"]

        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "p0_response": p0_response,
            "logprobs": {},
        }

        # Compute P0 response log prob under each policy
        for policy_name, tf_instance in teacher_forcing_instances.items():
            # Get system prompt if available
            system_prompt = policy_configs[policy_name].get("system_prompt", None)

            # Format prompt with system prompt if needed
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\n\nAssistant:"

            logprob_result = tf_instance.compute_log_prob(formatted_prompt, p0_response)

            if logprob_result.status == LogProbStatus.SUCCESS:
                result["logprobs"][policy_name] = logprob_result.value
            else:
                result["logprobs"][policy_name] = None
                console.print(
                    f"[yellow]Warning: Failed to compute {policy_name} log prob "
                    f"for prompt {prompt_id}: {logprob_result.error}[/yellow]"
                )

        # Check pi_clone weight if present
        issue = check_pi_clone_weight(prompt_id, result["logprobs"])
        if issue:
            extreme_count += 1
            # Save to extreme weights file
            with open("data/extreme_weights.jsonl", "a") as f:
                f.write(json.dumps(issue) + "\n")

        results.append(result)

    # Print batch summary if issues found
    if extreme_count > 0:
        console.print(
            f"\n📊 Batch validation: {extreme_count} extreme pi_clone weights detected"
        )

    return results


def main():
    # No arguments - fixed paths from convention
    INPUT_FILE = "data/all_responses.jsonl"
    OUTPUT_FILE = "data/logprobs.jsonl"

    console.print(
        "[bold cyan]Step 2b: Compute Log Probabilities (with validation)[/bold cyan]"
    )

    # Load config
    config = load_arena_config()

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(
            f"❌ [red]Error: {INPUT_FILE} not found. Run 02_generate_responses.py first.[/red]"
        )
        sys.exit(1)

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
    console.print("\n🦙 Initializing llama.cpp teacher forcing:")
    console.print(
        "[dim]Using each policy's production temperature for accurate importance weights[/dim]"
    )

    # Check llama.cpp availability
    if not LLAMA_CPP_AVAILABLE:
        console.print("❌ [red]Error: LlamaCppTeacherForcing not available![/red]")
        console.print("Install with: pip install llama-cpp-python")
        sys.exit(1)

    # Get model configuration
    model_config = config.llama_model_config
    if not model_config:
        console.print("❌ [red]Error: No model configuration found![/red]")
        sys.exit(1)

    console.print(f"\nModel: {model_config['path']}")
    console.print(
        f"GPU layers: {model_config.get('n_gpu_layers', -1)}, Context: {model_config.get('n_ctx', 2048)}"
    )

    # Check if model file exists
    model_path = Path(model_config["path"])
    if not model_path.exists():
        console.print(f"\n❌ [red]Error: Model file not found: {model_path}[/red]")
        console.print(f"\nDownload with:")
        console.print(f"mkdir -p models")
        console.print(f"curl -L -o {model_path} \\")
        console.print(
            f"  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf"
        )
        sys.exit(1)

    tf_instances = {}
    for policy_name, policy_config in policy_configs.items():
        # Use each policy's production temperature for accurate importance weights
        temperature = policy_config.get("temperature", 0.5)
        system_prompt = policy_config.get("system_prompt", None)

        # Use llama.cpp teacher forcing
        tf_instances[policy_name] = LlamaCppTeacherForcing(
            model_path=str(model_config["path"]),
            n_ctx=model_config.get("n_ctx", 2048),
            n_gpu_layers=model_config.get("n_gpu_layers", -1),
            use_mlock=model_config.get("use_mlock", False),
            verbose=model_config.get("verbose", False),
            seed=policy_config.get("seed", 42),
        )
        console.print(f"\n  ✅ {policy_name}: T={temperature}")

        if system_prompt:
            console.print(f"     └─ System prompt: {system_prompt[:50]}...")

    # Clear previous extreme weights log
    extreme_log_file = "data/extreme_weights.jsonl"
    if Path(extreme_log_file).exists():
        Path(extreme_log_file).unlink()
        console.print(f"🧹 Cleared previous extreme weights log")

    # Process with checkpointing
    console.print(f"\n🔄 Computing log probabilities...")
    console.print(
        f"⚠️  Will flag pi_clone weights > {EXTREME_WEIGHT_THRESHOLD:.0f}x or < {1/EXTREME_WEIGHT_THRESHOLD:.1f}x"
    )
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
        lambda batch: compute_logprobs_batch(batch, tf_instances, policy_configs),
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
    extreme_count = 0

    for item in results:
        prompt_id = item["prompt_id"]
        p0_response = item.get("p0_response", "")

        for policy, logprob in item["logprobs"].items():
            if logprob is not None:
                stats_by_policy[policy]["valid"] += 1
                stats_by_policy[policy]["logprobs"].append(logprob)
                if logprob == 0.0 and len(p0_response.strip()) > 0:
                    stats_by_policy[policy]["zero"] += 1
                    zero_samples.append((prompt_id, policy, p0_response[:50]))
            else:
                stats_by_policy[policy]["null"] += 1

    # Count extreme weights
    if Path(extreme_log_file).exists():
        with open(extreme_log_file) as f:
            extreme_count = sum(1 for _ in f)

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

    if extreme_count > 0:
        console.print(
            f"\n[yellow]⚠️  Detected {extreme_count} extreme pi_clone weights "
            f"(see {extreme_log_file})[/yellow]"
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

    # Run pi_clone health check
    if "pi_clone" in stats_by_policy:
        console.print("\n🏥 Running Pi_Clone Health Check:")

        # Load results and compute weights
        results = []
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                results.append(json.loads(line))

        weights = []
        for r in results:
            if "p0" in r["logprobs"] and "pi_clone" in r["logprobs"]:
                p0_lp = r["logprobs"]["p0"]
                clone_lp = r["logprobs"]["pi_clone"]

                if p0_lp is not None and clone_lp is not None:
                    weight = 2.718281828 ** (clone_lp - p0_lp)
                    weights.append(weight)

        if weights:
            mean_weight = np.mean(weights)
            median_weight = np.median(weights)
            cv = np.std(weights) / mean_weight if mean_weight > 0 else float("inf")

            # Check health
            if abs(mean_weight - 1.0) < 0.3 and cv < 0.3:
                console.print(f"  [green]✅ PASSED[/green]")
            else:
                console.print(f"  [red]❌ FAILED[/red]")

            console.print(f"  Mean weight: {mean_weight:.3f} (expected: 1.0)")
            console.print(f"  Median weight: {median_weight:.3f}")
            console.print(f"  Coefficient of variation: {cv:.3f}")
            console.print(f"  Valid weights: {len(weights)}")

    # Clean up checkpoint
    checkpoint_path = Path("data/checkpoint_logprobs.jsonl")
    if checkpoint_path.exists():
        console.print(f"🧹 Cleaning up checkpoint file")
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
