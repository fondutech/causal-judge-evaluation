#!/usr/bin/env python3
"""
Phase 1 - Step 4b: Add judge scores WITH UNCERTAINTY (structured output) to logging policy responses.

This script uses structured output uncertainty quantification.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from cje.utils.checkpointing import CheckpointManager as CM
from rich.console import Console
from rich.progress import track

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge.factory import JudgeFactory
from cje.utils.checkpointing import CheckpointManager, BatchProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add judge scores WITH UNCERTAINTY (structured output) to logging policy responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/p0_replies.jsonl",
        help="Input file with responses from Step 2",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/p0_scored_uncertainty.jsonl",
        help="Output file for scored responses",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/llama4-scout-instruct-basic",
        help="Judge model",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for judge API calls",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Dataset Preparation - Step 4b: UNCERTAINTY Judge Scores[/bold blue]"
    )
    console.print(f"📊 Scores will include variance from confidence intervals")
    console.print(f"📈 Using confidence interval uncertainty method")

    # Load responses
    console.print(f"📄 Loading responses from {args.input}")
    with open(args.input) as f:
        responses = [json.loads(line) for line in f]
    console.print(f"📊 Loaded {len(responses)} responses")

    # Create judge with uncertainty
    console.print(f"🔧 Creating uncertainty-aware judge with model: {args.model}")
    judge = JudgeFactory.create(
        model=args.model,
        provider="fireworks",
        temperature=0.0,
        uncertainty_method="confidence_interval",  # Uses CI for uncertainty
    )

    # Set up checkpointing
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.jsonl")
    checkpoint_mgr: CM = CheckpointManager(
        checkpoint_path=str(checkpoint_path), get_uid_fn=lambda x: x["prompt_id"]
    )

    # Process in batches with checkpointing
    processor = BatchProcessor(
        batch_size=args.batch_size, checkpoint_manager=checkpoint_mgr
    )

    def score_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch of responses."""
        results = []
        for response in batch:
            try:
                # Judge the response
                score = judge.score(
                    context=response["prompt"],
                    response=response["response"],
                )

                # Add score to response
                scored = response.copy()
                scored["judge_score"] = {
                    "mean": score.mean,
                    "variance": score.variance,
                }
                results.append(scored)
            except Exception as e:
                console.print(
                    f"[red]❌ Error scoring {response.get('prompt_id', 'unknown')}: {e}[/red]"
                )
                # Skip this response
        return results

    # Process all responses
    console.print(f"🔬 Scoring responses...")
    scored_responses = processor.process_batches(
        responses, score_batch, description="Scoring with uncertainty judge"
    )

    # Save final output
    console.print(
        f"\n💾 Saving {len(scored_responses)} scored responses to {args.output}"
    )
    with open(args.output, "w") as f:
        for response in scored_responses:
            f.write(json.dumps(response) + "\n")

    console.print(f"[green]✅ Successfully scored all responses![/green]")


if __name__ == "__main__":
    main()
