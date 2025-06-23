#!/usr/bin/env python3
"""
Phase 1 - Step 4b: Add judge scores WITH UNCERTAINTY (95% CI) to logging policy responses.

This script uses confidence interval based uncertainty quantification.
"""

import argparse
from add_judge_scores import main as score_main, console


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add judge scores WITH UNCERTAINTY (95% CI) to logging policy responses",
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
        "--checkpoint", type=str, help="Checkpoint file for resumable scoring"
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for judge API calls"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/llama4-scout-instruct-basic",
        help="Judge model",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 recommended for CI consistency)",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Dataset Preparation - Step 4b: UNCERTAINTY Judge Scores[/bold blue]"
    )
    console.print(f"📊 Scores will include 95% confidence intervals")
    console.print(f"📈 Variance calculated from CI width")

    # Call the shared implementation with CI settings
    # Create a namespace object to pass to score_main
    score_args = argparse.Namespace(
        input=args.input,
        output=args.output,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        model=args.model,
        temperature=args.temperature,
        uncertainty_method="confidence_interval",
    )

    score_main(score_args)


if __name__ == "__main__":
    main()
