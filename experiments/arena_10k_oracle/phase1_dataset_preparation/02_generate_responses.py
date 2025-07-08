#!/usr/bin/env python3
"""
Step 2: Generate responses from all policies (P0 and targets).

This script generates responses for all configured policies and saves them
in a consolidated format. Uses checkpointing for resumability.
"""

import argparse
import json
import os
import sys
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console
from cje.utils import CheckpointManager, BatchProcessor
from config_loader import load_arena_config


class AsyncResponseGenerator:
    """Handles async API calls for response generation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.fireworks.ai/inference/v1/chat/completions"

    async def generate_single(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a single response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with session.post(self.base_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"API error {resp.status}: {error_text}")

            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    async def generate_batch(
        self,
        prompts: List[Dict[str, Any]],
        model: str,
        temperature: float,
        policy_name: str,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt_data in prompts:
                # Format prompt
                if user_template:
                    formatted_prompt = user_template.format(
                        context=prompt_data["prompt"]
                    )
                else:
                    formatted_prompt = prompt_data["prompt"]

                if system_prompt:
                    formatted_prompt = f"{system_prompt}\n\n{formatted_prompt}"

                task = self.generate_single(
                    session, formatted_prompt, model, temperature
                )
                tasks.append((prompt_data, task))

            # Process in batches to avoid rate limits
            results = []
            batch_size = 10

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                console.print(
                    f"   Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}"
                )

                batch_results = await asyncio.gather(
                    *[task[1] for task in batch], return_exceptions=True
                )

                for (prompt_data, _), result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        console.print(f"[red]Error: {result}[/red]")
                        continue

                    results.append(
                        {
                            "prompt_id": prompt_data["prompt_id"],
                            "prompt": prompt_data["prompt"],
                            "response": result,
                            "policy": policy_name,
                            "model": model,
                            "temperature": temperature,
                            "metadata": prompt_data.get("metadata", {}),
                        }
                    )

                # Small delay between batches
                if i + batch_size < len(tasks):
                    await asyncio.sleep(0.5)

            return results


def generate_responses_for_policy(
    prompts: List[Dict[str, Any]],
    policy_name: str,
    policy_config: Dict[str, Any],
    generator: AsyncResponseGenerator,
    checkpoint_mgr: CheckpointManager,
) -> List[Dict[str, Any]]:
    """Generate responses for a single policy with checkpointing."""

    # Process in batches with checkpointing
    processor = BatchProcessor(checkpoint_manager=checkpoint_mgr, batch_size=50)

    def process_batch(batch):
        return asyncio.run(
            generator.generate_batch(
                batch,
                model=policy_config["model_name"],
                temperature=policy_config["temperature"],
                policy_name=policy_name,
                system_prompt=policy_config.get("system_prompt"),
                user_template=policy_config.get("user_template"),
            )
        )

    results = processor.process_batches(
        prompts,
        process_batch,
        description=f"Generating {policy_name} responses",
    )

    return results


def main():
    # Minimal arguments - everything else from config
    parser = argparse.ArgumentParser(description="Generate responses for all policies")
    args = parser.parse_args()

    # Fixed paths
    INPUT_FILE = "data/arena_prompts_10k.jsonl"
    OUTPUT_FILE = "data/all_responses.jsonl"

    console.print("[bold cyan]Step 2: Generate All Responses[/bold cyan]")

    # Check API key
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        console.print("❌ [red]Error: FIREWORKS_API_KEY not set![/red]")
        sys.exit(1)

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(
            f"❌ [red]Error: {INPUT_FILE} not found. Run 01_prepare_data.py first.[/red]"
        )
        sys.exit(1)

    # Load config
    config = load_arena_config()

    # Load prompts
    console.print(f"\n📄 Loading prompts from {INPUT_FILE}")
    with open(INPUT_FILE) as f:
        prompts = [json.loads(line) for line in f]
    console.print(f"✅ Loaded {len(prompts)} prompts")

    # Initialize generator
    generator = AsyncResponseGenerator(api_key)

    # Generate responses for all policies
    all_results = {}

    # 1. P0 (logging policy)
    console.print(f"\n[bold]Generating P0 responses...[/bold]")
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
        console.print(f"\n[bold]Generating {policy_name} responses...[/bold]")

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
                    "model": result["model"],
                    "temperature": result["temperature"],
                    "metadata": result.get("metadata", {}),
                }

                # Add system_prompt and user_template for target policies
                if policy_name != "p0":
                    policy_config = next(
                        p for p in config.target_policies if p["name"] == policy_name
                    )
                    if "system_prompt" in policy_config:
                        responses_by_prompt[prompt_id]["responses"][policy_name][
                            "system_prompt"
                        ] = policy_config["system_prompt"]
                    if "user_template" in policy_config:
                        responses_by_prompt[prompt_id]["responses"][policy_name][
                            "user_template"
                        ] = policy_config["user_template"]

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


if __name__ == "__main__":
    main()
