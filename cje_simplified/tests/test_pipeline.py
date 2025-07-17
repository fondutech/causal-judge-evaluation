"""Simple end-to-end test of CJE pipeline."""

import json
import tempfile
from pathlib import Path

from cje_simplified import (
    PrecomputedSampler,
    CalibratedIPS,
    create_calibrated_rewards,
    Llama3TemplateConfig,
    convert_chat_for_teacher_forcing,
)


def test_basic_pipeline():
    """Test basic CJE pipeline flow."""

    # Create minimal test data (need at least 10 samples with oracle for calibration)
    data = []
    for i in range(15):
        data.append(
            {
                "prompt": f"What is {i}+{i}?",
                "response": str(2 * i),
                "judge_score": 5 + 4 * (i / 15),  # Scores from 5 to 9
                "oracle_label": (
                    0.5 + 0.4 * (i / 15) if i < 10 else None
                ),  # First 10 have oracle
                "base_policy_logprob": -10 - i,
                "target_logps": {
                    "pi_better": -8 - i,  # Better (higher log prob)
                    "pi_worse": -15 - i,  # Worse (lower log prob)
                },
            }
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
        temp_file = f.name

    try:
        # Step 1: Calibrate judge scores
        calibrated_data, stats = create_calibrated_rewards(temp_file)
        assert len(calibrated_data) == 15
        assert all("reward" in d for d in calibrated_data)
        print(f"✓ Judge calibration worked: {stats['n_oracle']} oracle samples")

        # Step 2: Load data
        sampler = PrecomputedSampler(calibrated_data)
        assert sampler.n_samples == 15
        assert sampler.target_policies == ["pi_better", "pi_worse"]
        print(f"✓ Data loaded: {sampler.n_samples} samples")

        # Step 3: Run estimation
        estimator = CalibratedIPS(sampler, k_folds=2)
        results = estimator.fit_and_estimate()

        assert len(results.estimates) == 2
        print(
            f"✓ Estimation complete: pi_better={results.estimates[0]:.3f}, pi_worse={results.estimates[1]:.3f}"
        )

        # Just check that we got reasonable estimates
        assert all(0 <= e <= 1 for e in results.estimates)

    finally:
        Path(temp_file).unlink()


def test_chat_conversion():
    """Test chat format conversion."""

    chat = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # Test with Llama 3 template
    config = Llama3TemplateConfig()
    prompt_only, prompt_plus = convert_chat_for_teacher_forcing(
        chat, template_config=config, use_tokenizer=False
    )

    assert prompt_only.endswith("assistant<|end_header_id|>\n")
    assert prompt_plus.endswith("Hi there!<|eot_id|>")
    assert len(prompt_plus) > len(prompt_only)
    print("✓ Chat conversion works")


if __name__ == "__main__":
    print("Running CJE pipeline tests...\n")

    test_basic_pipeline()
    print()
    test_chat_conversion()

    print("\nAll tests passed! ✨")
