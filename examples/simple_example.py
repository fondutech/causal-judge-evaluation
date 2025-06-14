"""
Simple example showing how to use CJE for policy evaluation.

This demonstrates the canonical way to run CJE experiments using the CLI.
"""

import os
import json
from pathlib import Path
import yaml  # type: ignore[import-untyped]
import subprocess
from typing import Dict, Any, Optional


def create_experiment_config() -> Path:
    """Create a simple experiment configuration."""
    config = {
        "paths": {
            "work_dir": "outputs/simple_example",
        },
        "dataset": {
            "name": "ChatbotArena",
            "split": "train",
            "sample_limit": 100,  # Start small for testing
        },
        "logging_policy": {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_new_tokens": 1024,
        },
        "target_policies": [
            {
                "name": "baseline",
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_new_tokens": 1024,
            },
            {
                "name": "chain_of_thought",
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.3,
                "max_new_tokens": 1024,
                "system_prompt": "Let's think step by step before answering.",
            },
            {
                "name": "gpt4_upgrade",
                "provider": "openai",
                "model_name": "gpt-4",
                "temperature": 0.3,
                "max_new_tokens": 1024,
            },
        ],
        "judge": {
            "provider": "openai",
            "model_name": "gpt-4",
            "template": "default",
            "temperature": 0.0,
            "max_tokens": 100,
        },
        "estimator": {
            "name": "DRCPO",
            "k": 5,
            "clip": 20.0,
            "calibrate_weights": True,
            "calibrate_outcome": True,
        },
    }

    # Save config to file
    config_path = Path("configs/simple_experiment.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


def run_experiment(config_path: Path) -> Optional[Dict[str, Any]]:
    """Run CJE experiment using the CLI."""
    # Run the experiment
    cmd = [
        "cje",
        "run",
        "--cfg-path",
        str(config_path.parent),
        "--cfg-name",
        config_path.stem,
    ]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running experiment:")
        print(result.stderr)
        return None

    print(result.stdout)

    # Load and return results
    work_dir = Path("outputs/simple_example")
    result_file = work_dir / "result.json"

    if result_file.exists():
        with open(result_file) as f:
            data: Dict[str, Any] = json.load(f)
            return data

    return None


def analyze_results(results: Optional[Dict[str, Any]]) -> None:
    """Analyze and display experiment results."""
    if not results:
        print("No results to analyze")
        return

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    # Extract key information
    policy_rankings = results.get("policy_rankings", [])

    print("\nPolicy Rankings:")
    for i, policy in enumerate(policy_rankings, 1):
        name = policy["name"]
        estimate = policy["estimate"]
        ci_lower = policy["ci_lower"]
        ci_upper = policy["ci_upper"]

        print(f"{i}. {name}: {estimate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Check statistical significance
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS")
    print("=" * 60)

    if len(policy_rankings) > 1:
        baseline = policy_rankings[0]
        for policy in policy_rankings[1:]:
            # Check if CIs overlap
            overlap = (
                policy["ci_lower"] <= baseline["ci_upper"]
                and policy["ci_upper"] >= baseline["ci_lower"]
            )

            diff = policy["estimate"] - baseline["estimate"]

            print(f"\n{policy['name']} vs {baseline['name']}:")
            print(f"  Difference: {diff:+.4f}")
            print(
                f"  Significant: {'No (CIs overlap)' if overlap else 'Yes (CIs do not overlap)'}"
            )

    # Display metadata
    metadata = results.get("metadata", {})
    print(f"\nSample size: {metadata.get('sample_size', 'N/A')}")
    print(f"Analysis type: {metadata.get('analysis_type', 'N/A')}")


def main() -> None:
    """Run a simple CJE evaluation comparing different policies."""

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='sk-...'")
        return

    print("Creating experiment configuration...")
    config_path = create_experiment_config()
    print(f"Configuration saved to: {config_path}")

    print("\nRunning CJE experiment...")
    results = run_experiment(config_path)

    if results:
        analyze_results(results)
        print(f"\nFull results saved to: outputs/simple_example/result.json")
    else:
        print("Experiment failed - check error messages above")


if __name__ == "__main__":
    main()
