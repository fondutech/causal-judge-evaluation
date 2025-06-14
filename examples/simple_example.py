"""
Simple example showing how easy CJE can be to use.

This replaces complex multi-file examples with a single, clear script.
"""

from cje.config.simple import CJEConfig, PolicyConfig, DatasetConfig, EstimatorConfig
from cje.core import CJEPipeline
import os

# Make sure you have API keys set
# export OPENAI_API_KEY="sk-..."


def main() -> None:
    """Run a simple CJE evaluation comparing two prompts."""

    # 1. Define your experiment
    config = CJEConfig(
        # What generated your logged data?
        logging_policy=PolicyConfig(
            name="production",
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.3,
        ),
        # What policies do you want to evaluate?
        target_policies=[
            PolicyConfig(
                name="baseline",
                provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.3,
            ),
            PolicyConfig(
                name="chain_of_thought",
                provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                system_prompt="Let's think step by step before answering.",
            ),
            PolicyConfig(
                name="gpt4_upgrade",
                provider="openai",
                model_name="gpt-4",
                temperature=0.3,
            ),
        ],
        # Dataset settings
        dataset=DatasetConfig(
            name="ChatbotArena", sample_limit=100  # Start small for testing
        ),
        # Estimator settings
        estimator=EstimatorConfig(
            name="DRCPO",  # Doubly-robust estimator
            k_folds=5,  # Cross-validation folds
            clip=20.0,  # Clip importance weights at 20
        ),
        # Output directory
        work_dir="outputs/simple_example",
    )

    # 2. Run the pipeline
    pipeline = CJEPipeline(config)
    results = pipeline.run()

    # 3. Results are automatically displayed and saved
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(results.summary())

    # 4. Check if any policy is significantly better
    print("\n" + "=" * 60)
    print("POLICY COMPARISONS")
    print("=" * 60)

    baseline_idx = 0  # First policy is baseline
    for i in range(1, len(results.policy_names)):
        # Get estimates and standard errors
        baseline_est = results.estimates[baseline_idx]
        baseline_se = results.std_errors[baseline_idx]

        policy_est = results.estimates[i]
        policy_se = results.std_errors[i]

        # Compute difference and standard error of difference
        diff = policy_est - baseline_est
        # Assume independence for simplicity
        diff_se = (baseline_se**2 + policy_se**2) ** 0.5

        # Check if significant at 95% level
        is_significant = abs(diff) > 1.96 * diff_se

        print(f"\n{results.policy_names[i]} vs {results.policy_names[baseline_idx]}:")
        print(f"  Difference: {diff:+.4f} ± {diff_se:.4f}")
        print(f"  Significant: {'Yes' if is_significant else 'No'}")

        if results.ess_percentage is not None:
            print(f"  ESS: {results.ess_percentage[i]:.1f}%")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("export OPENAI_API_KEY='sk-...'")
        exit(1)

    main()
