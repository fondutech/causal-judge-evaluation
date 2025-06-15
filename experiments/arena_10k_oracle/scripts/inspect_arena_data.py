#!/usr/bin/env python3
"""
Quick script to inspect ChatBot Arena dataset structure and verify extraction logic.
"""

import json
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def inspect_dataset() -> None:
    """Load and inspect a few samples from the ChatBot Arena dataset."""

    console.print("📥 [bold blue]Loading ChatBot Arena dataset...[/bold blue]")

    # Load just a few samples to inspect
    ds = load_dataset("agie-ai/lmsys-chatbot_arena_conversations", split="train[:5]")

    console.print(f"\n✅ Loaded {len(ds)} samples for inspection")
    console.print(f"\n📊 Dataset features: {list(ds.features.keys())}")

    # Inspect each sample
    for idx, row in enumerate(ds):
        console.print(f"\n{'='*80}")
        console.print(f"[bold cyan]Sample {idx + 1}[/bold cyan]")
        console.print(f"{'='*80}")

        # Basic metadata
        console.print(f"\n[bold]Metadata:[/bold]")
        console.print(f"  Question ID: {row.get('question_id', 'N/A')}")
        console.print(f"  Model A: {row.get('model_a', 'N/A')}")
        console.print(f"  Model B: {row.get('model_b', 'N/A')}")
        console.print(f"  Winner: {row.get('winner', 'N/A')}")
        console.print(f"  Language: {row.get('language', 'N/A')}")
        console.print(f"  Turn: {row.get('turn', 'N/A')}")
        console.print(f"  Anonymous: {row.get('anony', 'N/A')}")

        # Check conversation structure
        conv_a = row.get("conversation_a", [])
        conv_b = row.get("conversation_b", [])

        console.print(f"\n[bold]Conversation A:[/bold] {len(conv_a)} messages")
        if conv_a:
            for i, msg in enumerate(conv_a[:2]):  # First 2 messages
                console.print(f"  Message {i+1}:")
                console.print(f"    Role: {msg.get('role', 'N/A')}")
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:97] + "..."
                console.print(f"    Content: {content}")

        console.print(f"\n[bold]Conversation B:[/bold] {len(conv_b)} messages")
        if conv_b:
            for i, msg in enumerate(conv_b[:2]):  # First 2 messages
                console.print(f"  Message {i+1}:")
                console.print(f"    Role: {msg.get('role', 'N/A')}")
                content = msg.get("content", "")
                if len(content) > 100:
                    content = content[:97] + "..."
                console.print(f"    Content: {content}")

        # Extract and show the user prompt (what our script would extract)
        console.print(f"\n[bold green]Extracted User Prompt:[/bold green]")
        conversation = conv_a or conv_b
        if conversation:
            for msg in conversation:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "").strip()
                    if len(prompt) > 200:
                        prompt = prompt[:197] + "..."
                    console.print(f"  {prompt}")
                    break
        else:
            console.print("  [red]No conversation found![/red]")

    # Test our extraction logic
    console.print(f"\n{'='*80}")
    console.print(
        "[bold yellow]Testing extraction logic on full dataset (first 100 samples)...[/bold yellow]"
    )
    console.print(f"{'='*80}\n")

    ds_larger = load_dataset(
        "agie-ai/lmsys-chatbot_arena_conversations", split="train[:100]"
    )

    prompts = []
    seen_prompts = set()
    empty_convs = 0
    no_user_msg = 0

    for idx, row in enumerate(ds_larger):
        # Use our extraction logic
        conversation = row.get("conversation_a") or row.get("conversation_b")
        if not conversation:
            empty_convs += 1
            continue

        first_turn = None
        for msg in conversation:
            if msg.get("role") == "user":
                first_turn = msg
                break

        if not first_turn:
            no_user_msg += 1
            continue

        prompt_text = first_turn.get("content", "").strip()

        if prompt_text and prompt_text not in seen_prompts:
            seen_prompts.add(prompt_text)
            prompts.append(
                {
                    "prompt_id": f"arena_{idx}",
                    "prompt": prompt_text,
                    "metadata": {
                        "question_id": row.get("question_id", f"q_{idx}"),
                        "model_a": row.get("model_a", "unknown"),
                        "model_b": row.get("model_b", "unknown"),
                        "winner": row.get("winner", "unknown"),
                    },
                }
            )

    console.print(
        f"✅ Successfully extracted {len(prompts)} unique prompts from {len(ds_larger)} samples"
    )
    console.print(f"📊 Empty conversations: {empty_convs}")
    console.print(f"📊 No user message found: {no_user_msg}")
    console.print(
        f"📊 Duplicate prompts filtered: {len(ds_larger) - empty_convs - no_user_msg - len(prompts)}"
    )

    # Show a few extracted prompts
    console.print(f"\n[bold]Sample extracted prompts:[/bold]")
    for i, p in enumerate(prompts[:3]):
        console.print(f"\n{i+1}. Prompt ID: {p['prompt_id']}")
        console.print(
            f"   Models: {p['metadata']['model_a']} vs {p['metadata']['model_b']}"
        )
        console.print(f"   Winner: {p['metadata']['winner']}")
        prompt_preview = (
            p["prompt"][:150] + "..." if len(p["prompt"]) > 150 else p["prompt"]
        )
        console.print(f"   Text: {prompt_preview}")


if __name__ == "__main__":
    try:
        inspect_dataset()
    except Exception as e:
        console.print(f"\n❌ [red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
