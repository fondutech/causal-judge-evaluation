#!/usr/bin/env python3
"""
Step 2: Generate responses from all policies using llama.cpp.

This script generates responses for all configured policies and saves them
in a consolidated format. Uses llama.cpp for deterministic generation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console
from cje.utils import CheckpointManager, BatchProcessor
from config_loader import load_arena_config

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LlamaResponseGenerator:
    """Handles response generation using llama.cpp."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        """Initialize llama.cpp model."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not installed")

        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            seed=42,  # Fixed seed for determinism
        )

    def generate_response(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int = 150,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single response."""
        # Format the prompt with system message if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\n\nAssistant:"

        # Generate response
        output = self.model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:", "\n\n"],
            echo=False,
        )

        return output["choices"][0]["text"].strip()

    def generate_batch(
        self,
        prompts: List[Dict[str, Any]],
        temperature: float,
        policy_name: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts."""
        results = []

        for i, prompt_data in enumerate(prompts):
            console.print(f"  Generating {policy_name} response {i+1}/{len(prompts)}")

            try:
                response = self.generate_response(
                    prompt=prompt_data["prompt"],
                    temperature=temperature,
                    system_prompt=system_prompt,
                )

                results.append(
                    {
                        "prompt_id": prompt_data["prompt_id"],
                        "prompt": prompt_data["prompt"],
                        "response": response,
                        "policy": policy_name,
                        "temperature": temperature,
                        "metadata": prompt_data.get("metadata", {}),
                    }
                )

            except Exception as e:
                console.print(f"[red]Error generating response: {str(e)[:100]}[/red]")
                # Add empty response to maintain alignment
                results.append(
                    {
                        "prompt_id": prompt_data["prompt_id"],
                        "prompt": prompt_data["prompt"],
                        "response": "[GENERATION_FAILED]",
                        "policy": policy_name,
                        "temperature": temperature,
                        "metadata": prompt_data.get("metadata", {}),
                    }
                )

        return results


def generate_responses_for_policy(
    prompts: List[Dict[str, Any]],
    policy_name: str,
    policy_config: Dict[str, Any],
    generator: LlamaResponseGenerator,
    checkpoint_mgr: CheckpointManager,
) -> List[Dict[str, Any]]:
    """Generate responses for a single policy with checkpointing."""

    # Process in batches with checkpointing
    processor = BatchProcessor(checkpoint_manager=checkpoint_mgr, batch_size=10)

    current_batch_num = [0]  # Use list to make it mutable in closure

    def process_batch(batch):
        current_batch_num[0] += 1
        batch_start = (current_batch_num[0] - 1) * 10 + 1
        batch_end = min(current_batch_num[0] * 10, len(prompts))
        console.print(
            f"\n   📦 Batch {current_batch_num[0]}/{(len(prompts) + 9) // 10}: "
            f"prompts {batch_start}-{batch_end}"
        )

        results = generator.generate_batch(
            batch,
            temperature=policy_config.get("temperature", 0.5),
            policy_name=policy_name,
            system_prompt=policy_config.get("system_prompt"),
        )

        return results

    results = processor.process_batches(
        prompts,
        process_batch,
        description=f"Generating {policy_name} responses",
    )

    return results


def main():
    # Fixed paths
    INPUT_FILE = "data/arena_prompts_10k.jsonl"
    OUTPUT_FILE = "data/all_responses.jsonl"

    console.print("[bold cyan]Step 2: Generate All Responses (llama.cpp)[/bold cyan]")

    # Check llama.cpp availability
    if not LLAMA_CPP_AVAILABLE:
        console.print("❌ [red]Error: llama-cpp-python not installed![/red]")
        sys.exit(1)

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(
            f"❌ [red]Error: {INPUT_FILE} not found. Run 01_prepare_data.py first.[/red]"
        )
        sys.exit(1)

    # Load config
    config = load_arena_config()

    # Check model file
    model_config = config.llama_model_config
    model_path = Path(model_config["path"])
    if not model_path.exists():
        console.print(f"❌ [red]Error: Model file not found: {model_path}[/red]")
        sys.exit(1)

    # Load prompts
    console.print(f"\n📄 Loading prompts from {INPUT_FILE}")
    with open(INPUT_FILE) as f:
        prompts = [json.loads(line) for line in f]
    console.print(f"✅ Loaded {len(prompts)} prompts")

    # Initialize generator
    console.print(f"\n🦙 Initializing llama.cpp model...")
    console.print(f"   Model: {model_path.name}")
    console.print(f"   GPU layers: {model_config.get('n_gpu_layers', -1)}")

    generator = LlamaResponseGenerator(
        model_path=str(model_path),
        n_ctx=model_config.get("n_ctx", 2048),
        n_gpu_layers=model_config.get("n_gpu_layers", -1),
    )

    # Generate responses for all policies
    all_results = {}

    # 1. P0 (logging policy)
    console.print(f"\n{'='*60}\n[bold cyan]P0 (baseline) policy[/bold cyan]\n{'='*60}")
    checkpoint_mgr = CheckpointManager(
        checkpoint_path="data/checkpoint_p0.jsonl",
        get_uid_fn=lambda x: x["prompt_id"],
    )

    p0_results = generate_responses_for_policy(
        prompts,
        "p0",
        config.logging_policy,
        generator,
        checkpoint_mgr,
    )
    all_results["p0"] = p0_results

    # Clean up P0 checkpoint
    if Path("data/checkpoint_p0.jsonl").exists():
        Path("data/checkpoint_p0.jsonl").unlink()

    # 2. Target policies
    for policy_config in config.target_policies:
        policy_name = policy_config["name"]
        console.print(
            f"\n{'='*60}\n[bold cyan]{policy_name} policy[/bold cyan]\n{'='*60}"
        )

        checkpoint_mgr = CheckpointManager(
            checkpoint_path=f"data/checkpoint_{policy_name}.jsonl",
            get_uid_fn=lambda x: x["prompt_id"],
        )

        target_results = generate_responses_for_policy(
            prompts,
            policy_name,
            policy_config,
            generator,
            checkpoint_mgr,
        )
        all_results[policy_name] = target_results

        # Clean up checkpoint
        checkpoint_file = Path(f"data/checkpoint_{policy_name}.jsonl")
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    # Create consolidated output
    console.print("\n📊 Creating consolidated responses file...")

    # Group by prompt_id
    responses_by_prompt = {}
    for prompt in prompts:
        prompt_id = prompt["prompt_id"]
        responses_by_prompt[prompt_id] = {
            "prompt_id": prompt_id,
            "prompt": prompt["prompt"],
            "metadata": prompt.get("metadata", {}),
            "responses": {},
        }

    # Add all responses
    for policy_name, results in all_results.items():
        for result in results:
            prompt_id = result["prompt_id"]
            if prompt_id in responses_by_prompt:
                responses_by_prompt[prompt_id]["responses"][policy_name] = {
                    "response": result["response"],
                    "temperature": result["temperature"],
                    "metadata": result.get("metadata", {}),
                }

                # Add system_prompt for policies that have it
                if policy_name != "p0":
                    policy_config = next(
                        (p for p in config.target_policies if p["name"] == policy_name),
                        None,
                    )
                    if policy_config and "system_prompt" in policy_config:
                        responses_by_prompt[prompt_id]["responses"][policy_name][
                            "system_prompt"
                        ] = policy_config["system_prompt"]

    # Save consolidated file
    with open(OUTPUT_FILE, "w") as f:
        for item in responses_by_prompt.values():
            f.write(json.dumps(item) + "\n")

    console.print(f"\n✅ [bold green]Saved all responses to {OUTPUT_FILE}[/bold green]")

    # Print summary
    console.print("\n[bold]Response Summary:[/bold]")
    total_expected = len(prompts) * (1 + len(config.target_policies))
    total_generated = sum(len(results) for results in all_results.values())
    console.print(f"  Expected: {total_expected} responses")
    console.print(f"  Generated: {total_generated} responses")

    for policy_name, results in all_results.items():
        console.print(f"  {policy_name}: {len(results)} responses")

    if total_generated == total_expected:
        console.print("\n[green]✅ All responses generated successfully![/green]")
    else:
        console.print("\n[yellow]⚠️  Some responses may be missing[/yellow]")


if __name__ == "__main__":
    main()
