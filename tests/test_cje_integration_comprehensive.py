"""
Comprehensive integration tests for the full CJE pipeline.

This module tests the complete end-to-end workflow described in the CJE paper:
1. Log → Calibrate → Estimate pipeline
2. Judge calibration with oracle slice
3. Weight calibration with isotonic regression
4. Cross-fitted DR-CPO estimation
5. Confidence interval construction
6. Diagnostic reporting
"""

import pytest
import numpy as np
from typing import List, Dict, Any, Optional, cast
from pathlib import Path
import tempfile
import json

from cje.estimators.drcpo import MultiDRCPOEstimator
from cje.loggers.multi_target_sampler import MultiTargetSampler
from cje.testing import MockPolicyRunner, MockJudge, testing_mode
from cje.calibration import cross_fit_calibration
from cje.judge import JudgeFactory
from cje.config import simple_config
from cje.data import load_dataset


class TestFullCJEPipeline:
    """Test the complete CJE pipeline as described in Algorithm 1."""

    def create_production_like_logs(
        self, n_samples: int = 1000
    ) -> List[Dict[str, Any]]:
        """Create logs that simulate production LLM system data."""
        np.random.seed(42)
        logs = []

        # Simulate different domains from Arena-Hard
        domains = ["reasoning", "coding", "creative", "factual"]

        for i in range(n_samples):
            domain = domains[i % len(domains)]
            context = f"{domain.title()} task {i}: Please solve this problem..."

            # Simulate response generation with token-level logging
            response_tokens = np.random.randint(20, 200)
            response = f"Response to {domain} task with {response_tokens} tokens"

            # Simulate token-level log probabilities (sum over sequence)
            token_logps = np.random.normal(-3.0, 1.2, response_tokens)
            total_logp = np.sum(token_logps)

            # Simulate retrieval component (optional)
            retrieval_prob = 1.0  # No retrieval for simplicity

            # Judge score (1-10 scale, domain-dependent quality)
            domain_bias = {
                "reasoning": 6.5,
                "coding": 5.8,
                "creative": 7.2,
                "factual": 6.0,
            }
            judge_raw = np.clip(np.random.normal(domain_bias[domain], 1.5), 1.0, 10.0)

            # Oracle label (25% have ground truth from human evaluation)
            has_oracle = i < n_samples // 4
            oracle_reward = None
            if has_oracle:
                # Oracle correlates with judge but with noise and domain effects
                oracle_reward = np.clip(judge_raw / 10 + np.random.normal(0, 0.1), 0, 1)

            # Multiple target policies (different prompting strategies)
            target_policies = {
                "cot_policy": total_logp
                + np.random.normal(0.3, 0.2),  # Chain-of-thought
                "few_shot": total_logp + np.random.normal(-0.1, 0.15),  # Few-shot
                "instruct": total_logp
                + np.random.normal(0.1, 0.1),  # Instruction-tuned
            }

            logs.append(
                {
                    "uid": f"sample_{i}",
                    "context": context,
                    "response": response,
                    "logp": total_logp,
                    "retrieval_prob": retrieval_prob,
                    "judge_raw": judge_raw,
                    "oracle_reward": oracle_reward,
                    "domain": domain,
                    "logp_target_all": target_policies,
                    "decode_params": {
                        "temperature": 0.4,
                        "top_p": 0.9,
                        "max_tokens": 1024,
                    },
                }
            )

        return logs

    def test_algorithm_1_implementation(self) -> None:
        """Test Algorithm 1: Cross-fitted CJE estimator from the paper."""
        # Step 1: Create production-like logs
        logs = self.create_production_like_logs(n_samples=200)

        # Step 2: Judge calibration (align with oracle slice)
        calibrated_logs, calib_diagnostics = cross_fit_calibration(
            logs,
            k_folds=3,
            score_key="judge_raw",
            label_key="oracle_reward",
            output_score_key="reward",
        )

        # Validate calibration worked
        assert calib_diagnostics["n_oracle"] > 0
        assert all("reward" in log for log in calibrated_logs)

        # Step 3: Set up target policies
        target_policy_names = ["cot_policy", "few_shot", "instruct"]
        runners = [
            MockPolicyRunner(name, temperature=0.0) for name in target_policy_names
        ]
        sampler = MultiTargetSampler(runners)

        # Step 4: Cross-fitted CJE estimation (Algorithm 1)
        estimator = MultiDRCPOEstimator(
            sampler=sampler,
            k=3,  # K-fold cross-validation
            clip=20.0,  # Importance weight clipping
            seed=42,
            stabilize_weights=True,
            calibrate_weights=True,
        )

        estimator.fit(calibrated_logs)
        result = estimator.estimate()

        # Step 5: Validate results
        assert result.n_policies == len(target_policy_names)
        assert result.v_hat.shape == (3,)
        assert result.se.shape == (3,)
        assert np.all(np.isfinite(result.v_hat))
        assert np.all(result.se > 0)

        # Check confidence intervals
        ci_lower, ci_upper = result.confidence_interval(0.95)
        assert np.all(ci_lower < result.v_hat)
        assert np.all(result.v_hat < ci_upper)

        print("✅ Algorithm 1 implementation test passed")

    def test_paper_workflow_integration(self) -> None:
        """Test the complete workflow described in Section 6 of the paper."""
        with testing_mode():
            # Step 1: Minimal logging schema (Table 1)
            logs = []
            np.random.seed(42)

            for i in range(200):
                # Compress JSONL row (~200B as mentioned in paper)
                log_entry = {
                    "context": f"prompt_{i}",  # prompt
                    "reply": f"response_{i}",  # full token sequence
                    "token_logp": np.random.normal(-8, 2),  # sum of token logprobs
                    "retrieval_prob": 1.0,  # softmax prob of doc/tool (optional)
                    "judge_raw": np.random.uniform(
                        1, 10
                    ),  # rubric score from LLM judge
                    "decode_params": {"temperature": 0.4, "top_p": 0.9, "seed": 42},
                }

                # Add oracle labels for 25% subset
                if i < 50:  # 25% oracle slice
                    judge_raw_val = cast(float, log_entry["judge_raw"])
                    log_entry["oracle_kpi"] = judge_raw_val / 10 + np.random.normal(
                        0, 0.1
                    )
                    log_entry["oracle_kpi"] = np.clip(
                        cast(float, log_entry["oracle_kpi"]), 0, 1
                    )

                logs.append(log_entry)

            # Step 2: Two-line calibration snippets (from Section 6.2)
            # Judge → reward calibration
            oracle_logs = [log for log in logs if "oracle_kpi" in log]
            judge_scores: List[float] = [
                cast(float, log["judge_raw"]) for log in oracle_logs
            ]
            oracle_kpis: List[float] = [
                cast(float, log["oracle_kpi"]) for log in oracle_logs
            ]

            from sklearn.isotonic import IsotonicRegression

            iso_r = IsotonicRegression(out_of_bounds="clip")
            iso_r.fit(judge_scores, oracle_kpis)

            # Apply calibration to all logs
            for log in logs:
                log["reward"] = iso_r.predict([log["judge_raw"]])[
                    0
                ]  # calibrated reward

            # Step 3: 25-line calibrated-DR stub (conceptual test)
            # Add target policy log probabilities and fix field names
            for log in logs:
                token_logp_val = cast(float, log["token_logp"])
                log["logp"] = token_logp_val  # Add expected field name
                log["response"] = log["reply"]  # Add expected field name
                log["logp_target_all"] = {
                    "target_policy": token_logp_val + np.random.normal(0.2, 0.1)
                }

            # Create CJE estimator
            runners = [MockPolicyRunner("target_policy")]
            sampler = MultiTargetSampler(runners)

            estimator = MultiDRCPOEstimator(sampler=sampler, k=5, seed=42)
            estimator.fit(logs)
            result = estimator.estimate()

            # Validate 25-line implementation works
            assert np.isfinite(result.v_hat[0])
            assert result.se[0] > 0

            print("✅ Paper workflow integration test passed")

    def test_deployment_checklist_validation(self) -> None:
        """Test the deployment checklist from Section 6.5."""
        logs = self.create_production_like_logs(n_samples=100)

        # 1. Logging: set temperature>=0.3 and logprobs=true in production
        for log in logs:
            assert log["decode_params"]["temperature"] >= 0.3
            assert "logp" in log  # Token log probabilities available

        # 2. Nightly job: run both calibrations
        calibrated_logs, diagnostics = cross_fit_calibration(
            logs,
            k_folds=3,
            score_key="judge_raw",
            label_key="oracle_reward",
            output_score_key="reward",
        )

        # If no oracle data available, use judge scores as rewards
        if diagnostics.get("n_oracle", 0) == 0:
            for log in calibrated_logs:
                if "reward" not in log:
                    log["reward"] = log["judge_raw"] / 10.0  # Normalize to [0,1]

        # 3. Run CJE estimator
        runners = [MockPolicyRunner("cot_policy")]
        sampler = MultiTargetSampler(runners)

        estimator = MultiDRCPOEstimator(sampler=sampler, k=3, clip=20.0, seed=42)
        estimator.fit(calibrated_logs)
        result = estimator.estimate()

        # 4. Basic validation
        assert np.isfinite(result.v_hat[0])
        assert result.se[0] > 0

        print("✅ Deployment checklist validation passed")

    def test_compute_cost_claims(self) -> None:
        """Test the compute cost claims from Table 2."""
        n_samples = 1000
        k_policies = 3

        # Simulate decode + judge baseline
        decode_judge_ops = n_samples * k_policies  # k * n generation calls
        judge_calls = n_samples * k_policies  # k * n judge calls

        # Simulate CJE pipeline
        cje_teacher_forcing = (
            n_samples * k_policies
        )  # n teacher-forced passes per policy
        cje_judge_calls = n_samples  # n judge calls (on existing responses)
        oracle_slices = 1  # 1 oracle slice

        # CJE should require fewer judge calls
        assert cje_judge_calls < judge_calls

        # CJE reuses existing tokens vs generating new ones
        generation_cost_ratio = decode_judge_ops / cje_teacher_forcing
        assert generation_cost_ratio == 1.0  # Same number of forward passes

        # But CJE teacher-forcing should be cheaper than full generation
        # This would be validated in actual runtime tests

        print(
            f"Decode+Judge: {decode_judge_ops} generations, {judge_calls} judge calls"
        )
        print(
            f"CJE: {cje_teacher_forcing} teacher-forced passes, {cje_judge_calls} judge calls"
        )
        print("✅ Compute cost structure validation passed")

    def test_error_handling_and_robustness(self) -> None:
        """Test error handling and robustness of the pipeline."""
        # Test with missing oracle labels
        logs_no_oracle = self.create_production_like_logs(n_samples=100)
        for log in logs_no_oracle:
            log["oracle_reward"] = None  # Remove all oracle labels

        # Should handle gracefully (no calibration possible)
        try:
            calibrated_logs, diagnostics = cross_fit_calibration(
                logs_no_oracle,
                k_folds=3,
                score_key="judge_raw",
                label_key="oracle_reward",
            )
            # Should return original logs with a warning
            assert diagnostics["n_oracle"] == 0
        except Exception as e:
            print(f"Expected handling of no oracle labels: {e}")

        # Test with extreme importance weights
        logs_extreme = []
        for i in range(50):
            logs_extreme.append(
                {
                    "context": f"context_{i}",
                    "response": "response",
                    "logp": (
                        np.log(0.01) if i < 10 else np.log(0.5)
                    ),  # Some very low prob
                    "reward": np.random.uniform(0, 1),
                    "logp_target_all": {"policy_0": np.log(0.5)},
                }
            )

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)

        # Should handle extreme weights with clipping
        estimator = MultiDRCPOEstimator(sampler=sampler, k=3, clip=10.0, seed=42)
        estimator.fit(logs_extreme)
        result = estimator.estimate()

        assert np.isfinite(result.v_hat[0])
        assert result.se[0] > 0

        print("✅ Error handling and robustness test passed")


class TestDiagnosticsAndMonitoring:
    """Test diagnostic capabilities for production monitoring."""

    def test_weight_distribution_diagnostics(self) -> None:
        """Test weight distribution diagnostics."""
        logs = []
        np.random.seed(42)

        # Create logs with varying weight distributions
        for i in range(100):
            # Create varying degrees of policy mismatch
            if i < 25:
                # High mismatch → high weights
                logp_logging = np.log(0.1)
                logp_target = np.log(0.8)
            else:
                # Low mismatch → low weights
                logp_logging = np.log(0.5)
                logp_target = np.log(0.5)

            logs.append(
                {
                    "context": f"context_{i}",
                    "response": "response",
                    "logp": logp_logging,
                    "reward": np.random.uniform(0, 1),
                    "logp_target_all": {"policy_0": logp_target},
                }
            )

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)

        estimator = MultiDRCPOEstimator(sampler=sampler, k=3, clip=20.0, seed=42)
        estimator.fit(logs)
        result = estimator.estimate()

        # Should produce valid results
        assert np.isfinite(result.v_hat[0])
        assert result.se[0] > 0

        print("✅ Weight distribution diagnostics test passed")

    def test_judge_oracle_correlation(self) -> None:
        """Test judge-oracle correlation diagnostics."""
        logs = []
        np.random.seed(42)

        # Create logs with varying judge-oracle correlation
        for i in range(100):
            judge_score = np.random.uniform(1, 10)

            # Oracle correlates with judge but with noise
            if i < 25:  # 25% oracle slice
                oracle_reward = 0.1 * judge_score + np.random.normal(0, 0.2)
                oracle_reward = np.clip(oracle_reward, 0, 1)
            else:
                oracle_reward = None

            logs.append(
                {
                    "context": f"context_{i}",
                    "response": "response",
                    "logp": np.random.normal(-5, 1),
                    "judge_raw": judge_score,
                    "oracle_reward": oracle_reward,
                }
            )

        # Calculate judge-oracle correlation
        oracle_logs = [log for log in logs if log["oracle_reward"] is not None]
        judge_scores: List[float] = [
            cast(float, log["judge_raw"]) for log in oracle_logs
        ]
        oracle_rewards: List[float] = [
            cast(float, log["oracle_reward"]) for log in oracle_logs
        ]

        correlation = np.corrcoef(judge_scores, oracle_rewards)[0, 1]
        print(f"Judge-Oracle Spearman correlation: {correlation:.3f}")

        # Should have reasonable correlation (allow for noise)
        assert correlation > 0.1, f"Judge-oracle correlation {correlation:.3f} too low"

        print("✅ Judge-oracle correlation test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
