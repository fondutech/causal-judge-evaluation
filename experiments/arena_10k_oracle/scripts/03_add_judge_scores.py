#!/usr/bin/env python3
"""
Step 3: Add judge scores to logging policy responses.

This script:
1. Loads π₀ responses from Step 2
2. Scores each response using the same model as judge (Llama-3-34B at T=0)
3. Uses a 0-10 helpfulness/correctness/safety rubric
4. Adds raw judge scores for later calibration

Usage:
    python 03_add_judge_scores.py --input ../data/p0_replies.jsonl --output ../data/p0_scored.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge import JudgeFactory
from cje.utils.checkpointing import create_jsonl_checkpoint_manager, BatchProcessor
from cje.utils.progress import console


# Default judge rubric for Arena 10K experiment
DEFAULT_RUBRIC = """Rate the assistant's response on a scale of 0-10 considering:
- Helpfulness: Does it address the user's needs?
- Correctness: Is the information accurate?  
- Safety: Is the response appropriate and harmless?

Provide a single integer score from 0 to 10."""


def load_responses(input_path: str) -> List[Dict[str, Any]]:
    """Load responses from JSONL file."""
    responses = []
    with open(input_path, "r") as f:
        for line in f:
            responses.append(json.loads(line))

    console.print(f"📄 Loaded {len(responses):,} responses from {input_path}")
    return responses


def get_api_key() -> str:
    """Get Fireworks API key from environment or AWS Secrets Manager."""
    # First try environment variable
    env_key = os.getenv("FIREWORKS_API_KEY")
    if env_key:
        return env_key

    # Try AWS Secrets Manager
    try:
        from cje.utils.aws_secrets import get_api_key_from_secrets

        api_key = get_api_key_from_secrets(
            secret_name="cje/prod/api-keys",
            key="FIREWORKS_API_KEY",
            env_var_name="FIREWORKS_API_KEY",
            cache_in_env=True,
        )
        return str(api_key)

    except Exception:
        raise ValueError("No Fireworks API key available. Please set FIREWORKS_API_KEY")


def score_responses_with_judge(
    responses: List[Dict[str, Any]],
    judge_model: str = "accounts/fireworks/models/llama-v3-34b-instruct",
    temperature: float = 0.0,
    rubric: str = DEFAULT_RUBRIC,
    batch_size: int = 32,
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Score responses using LLM judge.

    Uses same model as logging policy but at T=0 for deterministic scoring.
    """
    # Setup checkpointing
    checkpoint_manager = create_jsonl_checkpoint_manager(checkpoint_path)

    # Create batch processor
    batch_processor = BatchProcessor(
        batch_size=batch_size,
        checkpoint_manager=checkpoint_manager,
        continue_on_error=True,
    )

    # Initialize judge
    console.print(f"🔬 [bold blue]Initializing judge[/bold blue]")
    console.print(f"📊 Model: {judge_model}")
    console.print(f"🌡️  Temperature: {temperature} (deterministic)")

    judge = JudgeFactory.create(
        provider="fireworks",
        model=judge_model,
        temperature=temperature,
        structured_output_schema="JudgeScore",  # Use basic score schema
    )

    def process_batch(batch_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of responses to add judge scores."""
        batch_results = []

        for response_data in batch_responses:
            try:
                # Create judge prompt
                prompt = response_data["prompt"]
                response = response_data["response"]

                # Format conversation for judge
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]

                # Score with judge
                score_result = judge.score(context=prompt, response=response)

                # Extract numeric score
                if isinstance(score_result, dict):
                    score = score_result.get("score", None)
                else:
                    # Try to parse as integer if string
                    try:
                        score = int(score_result)
                    except:
                        console.print(
                            f"⚠️  [yellow]Failed to parse score: {score_result}[/yellow]"
                        )
                        score = None

                # Add judge score to response data
                result = {
                    **response_data,
                    "judge_score_raw": score,
                    "judge": {
                        "model": judge_model,
                        "temperature": temperature,
                        "rubric": rubric,
                        "timestamp": time.time(),
                    },
                }

                batch_results.append(result)

            except Exception as e:
                console.print(f"⚠️  [yellow]Error scoring response: {e}[/yellow]")
                # Keep the response but mark as failed
                result = {
                    **response_data,
                    "judge_score_raw": None,
                    "judge_error": str(e),
                }
                batch_results.append(result)

        return batch_results

    # Process all responses with progress tracking
    start_time = time.time()

    try:
        results = batch_processor.process_batches(
            responses,
            process_batch,
            description="Scoring responses with judge",
            auto_save_frequency=1,  # Save after every batch
        )

        total_time = time.time() - start_time

        # Count successful scores
        successful = sum(1 for r in results if r.get("judge_score_raw") is not None)

        console.print(
            f"\n✅ [green]Scored {successful:,}/{len(results):,} responses[/green]"
        )
        console.print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        if successful > 0:
            console.print(f"📈 Average: {total_time/successful:.2f} seconds per score")

        return results

    except KeyboardInterrupt:
        console.print(
            f"\n⚠️ [yellow]Interrupted - Progress saved to checkpoint[/yellow]"
        )
        return checkpoint_manager.get_processed_items()
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        if checkpoint_manager:
            console.print(f"💾 Partial progress saved to: {checkpoint_path}")
        raise


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    console.print(
        f"💾 [green]Saved {len(results):,} scored responses to {output_file}[/green]"
    )

    # Show score distribution
    scores = [
        r["judge_score_raw"] for r in results if r.get("judge_score_raw") is not None
    ]
    if scores:
        avg_score = sum(scores) / len(scores)
        console.print(f"📊 Average judge score: {avg_score:.2f}")
        console.print(f"📊 Score range: {min(scores)} - {max(scores)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add judge scores to logging policy responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python 03_add_judge_scores.py --input ../data/p0_replies.jsonl
  
  # With checkpointing (recommended)
  python 03_add_judge_scores.py --input ../data/p0_replies.jsonl --checkpoint checkpoint.jsonl
  
  # Custom batch size for API efficiency
  python 03_add_judge_scores.py --input ../data/p0_replies.jsonl --batch-size 64
  
  # Resume interrupted job
  python 03_add_judge_scores.py --input ../data/p0_replies.jsonl --checkpoint checkpoint.jsonl
        """,
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
        default="../data/p0_scored.jsonl",
        help="Output file for scored responses",
    )

    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint file for resumable scoring"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for judge API calls"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/llama-v3-34b-instruct",
        help="Judge model (default: same as logging policy)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature (default: 0.0 for deterministic)",
    )

    parser.add_argument(
        "--rubric", type=str, default=DEFAULT_RUBRIC, help="Scoring rubric for judge"
    )

    parser.add_argument(
        "--cleanup-checkpoint",
        action="store_true",
        help="Delete checkpoint file on successful completion",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 3: Add Judge Scores[/bold blue]"
    )

    # Auto-enable checkpointing for large runs
    if not args.checkpoint:
        responses = load_responses(args.input)
        if len(responses) > 1000:
            args.checkpoint = args.output.replace(".jsonl", "_checkpoint.jsonl")
            console.print(f"📝 Auto-enabling checkpointing: {args.checkpoint}")

    try:
        # Get API key
        api_key = get_api_key()

        # Load responses
        responses = load_responses(args.input)

        # Score responses
        results = score_responses_with_judge(
            responses,
            judge_model=args.model,
            temperature=args.temperature,
            rubric=args.rubric,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint,
        )

        # Save results
        save_results(results, args.output)

        # Cleanup checkpoint if requested
        if args.checkpoint and os.path.exists(args.checkpoint):
            if args.cleanup_checkpoint:
                try:
                    os.remove(args.checkpoint)
                    console.print(f"🗑️  [green]Cleaned up checkpoint[/green]")
                except Exception as e:
                    console.print(
                        f"⚠️  [yellow]Could not delete checkpoint: {e}[/yellow]"
                    )
            else:
                console.print(f"🗑️  You can now delete checkpoint: {args.checkpoint}")

        console.print(f"\n✅ [bold green]Step 3 complete![/bold green]")
        console.print(f"📄 Output: {args.output}")

    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Interrupted by user[/yellow]")
        if args.checkpoint:
            console.print(f"💾 Resume with the same command")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
