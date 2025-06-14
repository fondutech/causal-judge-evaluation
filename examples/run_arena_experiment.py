#!/usr/bin/env python3
"""
Simple Arena CJE Experiment Runner
==================================

Basic script to run arena CJE experiments using the standard CJE pipeline.
"""

import argparse
from examples.arena_analysis_python import run_arena_experiment


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Arena CJE Experiment")
    parser.add_argument(
        "--config",
        default="arena_test",
        help="Configuration file to use (default: arena_test)",
    )

    args = parser.parse_args()

    # Run the experiment
    results = run_arena_experiment(args.config)

    print(f"✅ Experiment complete! Results: {results}")


if __name__ == "__main__":
    main()
