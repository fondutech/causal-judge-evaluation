#!/usr/bin/env python3
"""
Estimate costs for both sample and full runs based on current API pricing.
"""

import json
from pathlib import Path
from typing import Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class CostEstimator:
    """Estimate API costs for Arena 10K experiment."""

    def __init__(self):
        # Current API pricing (as of 2024)
        self.pricing = {
            # Fireworks pricing per 1M tokens
            "fireworks": {
                "llama4-scout-instruct-basic": {
                    "input": 0.20,  # $0.20 per 1M input tokens
                    "output": 0.20,  # $0.20 per 1M output tokens
                },
                "llama4-maverick-instruct-basic": {
                    "input": 0.40,
                    "output": 0.40,
                },
            },
            # OpenAI pricing
            "openai": {
                "gpt-4o": {
                    "input": 2.50,  # $2.50 per 1M input tokens
                    "output": 10.00,  # $10.00 per 1M output tokens
                },
                "gpt-3.5-turbo": {
                    "input": 0.50,
                    "output": 1.50,
                },
            },
        }

        # Estimated tokens per operation
        self.tokens_per_op = {
            "p0_generation": {"input": 500, "output": 200},
            "target_generation": {"input": 500, "output": 200},
            "teacher_forcing": {"input": 700, "output": 50},  # Mostly input
            "oracle_labeling": {"input": 800, "output": 100},
            "judge_scoring": {"input": 700, "output": 50},
        }

    def calculate_step_cost(
        self, step: str, model: str, provider: str, num_calls: int
    ) -> float:
        """Calculate cost for a specific step."""
        if provider not in self.pricing:
            return 0.0
        if model not in self.pricing[provider]:
            # Use first available model as proxy
            model = list(self.pricing[provider].keys())[0]

        tokens = self.tokens_per_op.get(step, {"input": 500, "output": 100})
        pricing = self.pricing[provider][model]

        # Calculate cost
        input_cost = (num_calls * tokens["input"] / 1_000_000) * pricing["input"]
        output_cost = (num_calls * tokens["output"] / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def estimate_phase1_costs(self, num_prompts: int) -> Dict[str, Any]:
        """Estimate costs for Phase 1."""
        costs = {}

        # Step 2a: P0 Generation
        costs["2a_p0_generation"] = {
            "calls": num_prompts,
            "model": "llama4-scout-instruct-basic",
            "provider": "fireworks",
            "cost": self.calculate_step_cost(
                "p0_generation", "llama4-scout-instruct-basic", "fireworks", num_prompts
            ),
        }

        # Step 2b: Target Generation (4 policies)
        target_policies = {
            "pi_clone": "llama4-scout-instruct-basic",
            "pi_cot": "llama4-scout-instruct-basic",
            "pi_bigger_model": "llama4-maverick-instruct-basic",
            "pi_bad": "llama4-scout-instruct-basic",
        }

        target_cost = 0
        for policy, model in target_policies.items():
            cost = self.calculate_step_cost(
                "target_generation", model, "fireworks", num_prompts
            )
            target_cost += cost

        costs["2b_target_generation"] = {
            "calls": num_prompts * 4,
            "models": list(target_policies.values()),
            "provider": "fireworks",
            "cost": target_cost,
        }

        # Step 2c: Teacher Forcing (4 policies)
        tf_cost = 0
        for policy, model in target_policies.items():
            cost = self.calculate_step_cost(
                "teacher_forcing", model, "fireworks", num_prompts
            )
            tf_cost += cost

        costs["2c_teacher_forcing"] = {
            "calls": num_prompts * 4,
            "models": list(target_policies.values()),
            "provider": "fireworks",
            "cost": tf_cost,
        }

        # Step 3: Oracle Labeling
        costs["3_oracle_labeling"] = {
            "calls": num_prompts,
            "model": "gpt-4o",
            "provider": "openai",
            "cost": self.calculate_step_cost(
                "oracle_labeling", "gpt-4o", "openai", num_prompts
            ),
        }

        # Step 4: Judge Scoring (5 responses per prompt)
        costs["4_judge_scoring"] = {
            "calls": num_prompts * 5,  # P0 + 4 targets
            "model": "llama4-scout-instruct-basic",
            "provider": "fireworks",
            "cost": self.calculate_step_cost(
                "judge_scoring",
                "llama4-scout-instruct-basic",
                "fireworks",
                num_prompts * 5,
            ),
        }

        # Total
        total_cost = sum(step["cost"] for step in costs.values())
        total_calls = sum(step["calls"] for step in costs.values())

        return {
            "steps": costs,
            "total_cost": total_cost,
            "total_calls": total_calls,
            "cost_per_prompt": total_cost / num_prompts if num_prompts > 0 else 0,
        }

    def display_estimates(self):
        """Display cost estimates for sample and full runs."""
        console.print(
            Panel.fit("Arena 10K Cost Estimation", title="💰 API Cost Calculator")
        )

        # Calculate for both sample and full
        sample_est = self.estimate_phase1_costs(100)
        full_est = self.estimate_phase1_costs(10000)

        # Display table
        table = Table(title="Phase 1 Cost Breakdown")
        table.add_column("Step", style="cyan")
        table.add_column("API Calls (Sample)", style="yellow", justify="right")
        table.add_column("Cost (Sample)", style="green", justify="right")
        table.add_column("API Calls (Full)", style="yellow", justify="right")
        table.add_column("Cost (Full)", style="green", justify="right")

        for step_name, step_data in sample_est["steps"].items():
            full_step = full_est["steps"][step_name]
            table.add_row(
                step_name.replace("_", " ").title(),
                f"{step_data['calls']:,}",
                f"${step_data['cost']:.3f}",
                f"{full_step['calls']:,}",
                f"${full_step['cost']:.2f}",
            )

        table.add_row(
            "TOTAL",
            f"{sample_est['total_calls']:,}",
            f"${sample_est['total_cost']:.3f}",
            f"{full_est['total_calls']:,}",
            f"${full_est['total_cost']:.2f}",
            style="bold",
        )

        console.print(table)

        # Additional insights
        console.print("\n📊 Key Insights:")
        console.print(f"  • Cost per prompt: ${full_est['cost_per_prompt']:.4f}")
        console.print(f"  • Sample run (1%): ${sample_est['total_cost']:.2f}")
        console.print(f"  • Full run (100%): ${full_est['total_cost']:.2f}")
        console.print(
            f"  • Scale factor: {full_est['total_cost'] / sample_est['total_cost']:.1f}x"
        )

        # Breakdown by provider
        console.print("\n🏢 Cost by Provider:")
        fireworks_cost = sum(
            s["cost"]
            for s in full_est["steps"].values()
            if s.get("provider") == "fireworks"
        )
        openai_cost = sum(
            s["cost"]
            for s in full_est["steps"].values()
            if s.get("provider") == "openai"
        )

        console.print(
            f"  • Fireworks: ${fireworks_cost:.2f} ({fireworks_cost/full_est['total_cost']*100:.1f}%)"
        )
        console.print(
            f"  • OpenAI: ${openai_cost:.2f} ({openai_cost/full_est['total_cost']*100:.1f}%)"
        )

        # Budget recommendations
        console.print("\n💡 Budget Recommendations:")
        console.print(
            f"  • Minimum budget: ${full_est['total_cost'] * 1.2:.2f} (20% buffer)"
        )
        console.print(
            f"  • Recommended: ${full_est['total_cost'] * 1.5:.2f} (50% buffer)"
        )
        console.print("  • Consider API rate limits and retries")

        return sample_est, full_est


def main():
    """Main entry point."""
    estimator = CostEstimator()
    estimator.display_estimates()


if __name__ == "__main__":
    main()
