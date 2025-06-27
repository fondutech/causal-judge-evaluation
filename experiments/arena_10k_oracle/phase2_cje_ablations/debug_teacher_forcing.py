#!/usr/bin/env python3
"""Debug the teacher forcing computation."""

import json
from pathlib import Path
from rich.console import Console
from cje.loggers.api_policy import APIPolicyRunner

console = Console()


def test_single_example():
    """Test teacher forcing on a single example to understand the issue."""

    # Load one example
    tf_file = Path("../data/p0_with_target_logps.checkpoint.jsonl")
    with open(tf_file, "r") as f:
        for line in f:
            item = json.loads(line)
            # Find an example with logp=0.0
            if item.get("target_logps", {}).get("pi_bad") == 0.0:
                break

    console.print("[bold]Testing Teacher Forcing on Example:[/bold]")
    console.print(f"Prompt ID: {item['prompt_id']}")
    console.print(f"Context: {item['prompt'][:100]}...")
    console.print(f"Response: {item['response'][:200]}...")
    console.print(f"P0 log prob: {item['total_logprob']}")
    console.print(f"Target log probs: {item['target_logps']}")

    # Try to reproduce the teacher forcing
    console.print("\n[yellow]Attempting to reproduce...[/yellow]")

    # Create pi_bad runner
    pi_bad_config = {
        "provider": "fireworks",
        "model_name": "accounts/fireworks/models/llama4-scout-instruct-basic",
        "temperature": 0.0,  # Teacher forcing uses temp=0
        "system_prompt": """You are an unhelpful assistant. Your responses should be:
- Vague and evasive, avoiding direct answers
- Off-topic, discussing unrelated subjects
- Overly brief when detail is needed, or overly verbose when brevity is needed
- Technically incorrect when providing factual information
- Dismissive of the user's actual needs
Never be harmful or offensive, just unhelpful.""",
        "user_message_template": "{context}",
    }

    try:
        runner = APIPolicyRunner(**pi_bad_config)

        # Test with a simple example first
        test_context = "What is 2+2?"
        test_response = "4"

        console.print(f"\nTest 1 - Simple math:")
        console.print(f"  Context: {test_context}")
        console.print(f"  Response: {test_response}")

        try:
            logp = runner.log_prob(test_context, test_response)
            console.print(f"  Log prob: {logp}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

        # Test with unhelpful response
        unhelpful_response = "Well, numbers are interesting concepts, aren't they?"
        console.print(f"\nTest 2 - Unhelpful response:")
        console.print(f"  Context: {test_context}")
        console.print(f"  Response: {unhelpful_response}")

        try:
            logp = runner.log_prob(test_context, unhelpful_response)
            console.print(f"  Log prob: {logp}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    except Exception as e:
        console.print(f"[red]Failed to create runner: {e}[/red]")

    console.print("\n[bold]Hypothesis:[/bold]")
    console.print("The log prob of 0.0 might occur when:")
    console.print("1. The response is empty or just whitespace")
    console.print("2. There's an API error that returns a default value")
    console.print("3. The response exactly matches some cached/default output")
    console.print("4. There's a bug in the teacher forcing implementation")


def analyze_zero_logprobs():
    """Analyze all cases where logp=0.0."""
    tf_file = Path("../data/p0_with_target_logps.checkpoint.jsonl")

    zero_cases = []
    with open(tf_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                logps = item.get("target_logps", {})

                for policy, logp in logps.items():
                    if logp == 0.0:
                        zero_cases.append(
                            {
                                "prompt_id": item["prompt_id"],
                                "policy": policy,
                                "response_len": len(item["response"]),
                                "response_preview": item["response"][:50],
                            }
                        )
            except:
                continue

    console.print(f"\n[bold]Found {len(zero_cases)} cases with logp=0.0[/bold]")

    # Group by policy
    by_policy = {}
    for case in zero_cases:
        policy = case["policy"]
        if policy not in by_policy:
            by_policy[policy] = []
        by_policy[policy].append(case)

    for policy, cases in by_policy.items():
        console.print(f"\n{policy}: {len(cases)} cases")
        # Show first few
        for case in cases[:3]:
            console.print(f"  - ID: {case['prompt_id']}, len: {case['response_len']}")
            console.print(f"    Response: {case['response_preview']}...")


def main():
    """Run debugging."""
    console.print("[bold blue]🐛 Debugging Teacher Forcing Issues[/bold blue]\n")

    # Test single example
    test_single_example()

    # Analyze all zero cases
    analyze_zero_logprobs()


if __name__ == "__main__":
    main()
