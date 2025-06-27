#!/usr/bin/env python3
"""Analyze why pi_bad still has high SNIPS scores."""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console()


def analyze_response_alignment():
    """Check if P0 responses align more with pi_bad's style."""

    # Load some P0 responses
    p0_file = Path("../data/p0_replies.jsonl")
    tf_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    # Analyze response characteristics
    response_lengths = []
    responses_sample = []

    with open(p0_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 100:  # Sample first 100
                break
            try:
                item = json.loads(line)
                response = item["response"]
                response_lengths.append(len(response))
                if i < 5:
                    responses_sample.append(
                        {
                            "prompt": item["prompt"][:80] + "...",
                            "response": response[:150] + "...",
                        }
                    )
            except:
                continue

    console.print("[bold]P0 Response Characteristics:[/bold]")
    console.print(f"Average length: {np.mean(response_lengths):.0f} chars")
    console.print(f"Median length: {np.median(response_lengths):.0f} chars")

    console.print("\n[bold]Sample P0 Responses:[/bold]")
    for i, sample in enumerate(responses_sample, 1):
        console.print(f"\n{i}. Prompt: {sample['prompt']}")
        console.print(f"   Response: {sample['response']}")

    # Analyze importance weights distribution
    console.print("\n[bold]Analyzing Importance Weight Patterns:[/bold]")

    # Count high-weight cases for each policy
    high_weight_cases = {"pi_bad": [], "pi_bigger_model": [], "pi_cot": []}

    with open(tf_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                p0_logp = item["total_logprob"]

                for policy, target_logp in item.get("target_logps", {}).items():
                    weight = np.exp(np.clip(target_logp - p0_logp, -20, 20))

                    if weight > 10:  # High weight cases
                        high_weight_cases[policy].append(
                            {
                                "prompt_id": item["prompt_id"],
                                "weight": weight,
                                "response_len": len(item["response"]),
                                "response_preview": item["response"][:100],
                            }
                        )
            except:
                continue

    # Show statistics
    table = Table(title="High Weight Cases (weight > 10)")
    table.add_column("Policy", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Max Weight", style="yellow")
    table.add_column("Avg Response Len", style="magenta")

    for policy, cases in high_weight_cases.items():
        if cases:
            max_weight = max(c["weight"] for c in cases)
            avg_len = np.mean([c["response_len"] for c in cases])
            table.add_row(
                policy, str(len(cases)), f"{max_weight:.0e}", f"{avg_len:.0f}"
            )
        else:
            table.add_row(policy, "0", "-", "-")

    console.print(table)

    # Show examples where pi_bad has high weight
    console.print("\n[bold]Examples where pi_bad has high weight:[/bold]")
    bad_cases = sorted(
        high_weight_cases["pi_bad"], key=lambda x: x["weight"], reverse=True
    )[:3]

    for i, case in enumerate(bad_cases, 1):
        console.print(f"\n{i}. ID: {case['prompt_id']}, Weight: {case['weight']:.0e}")
        console.print(f"   Response: {case['response_preview']}...")


def analyze_policy_preferences():
    """Analyze which types of responses each policy prefers."""
    tf_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    # Categorize responses by characteristics
    short_responses = []  # < 100 chars
    medium_responses = []  # 100-500 chars
    long_responses = []  # > 500 chars

    with open(tf_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                response_len = len(item["response"])

                if response_len < 100:
                    short_responses.append(item)
                elif response_len < 500:
                    medium_responses.append(item)
                else:
                    long_responses.append(item)
            except:
                continue

    console.print(f"\n[bold]Response Length Distribution:[/bold]")
    console.print(f"Short (<100): {len(short_responses)}")
    console.print(f"Medium (100-500): {len(medium_responses)}")
    console.print(f"Long (>500): {len(long_responses)}")

    # Analyze which policy prefers which length
    def analyze_category(responses, category_name):
        if not responses:
            return

        console.print(f"\n[yellow]{category_name} Responses:[/yellow]")

        avg_weights = {"pi_bad": [], "pi_bigger_model": [], "pi_cot": []}

        for item in responses:
            p0_logp = item["total_logprob"]
            for policy, target_logp in item.get("target_logps", {}).items():
                weight = np.exp(np.clip(target_logp - p0_logp, -20, 20))
                avg_weights[policy].append(weight)

        for policy, weights in avg_weights.items():
            if weights:
                console.print(f"  {policy}: avg weight = {np.mean(weights):.3f}")

    analyze_category(short_responses, "Short")
    analyze_category(medium_responses, "Medium")
    analyze_category(long_responses, "Long")


def check_unhelpful_alignment():
    """Check if P0 responses accidentally align with 'unhelpful' patterns."""
    tf_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    # Keywords that might indicate unhelpful responses
    unhelpful_patterns = [
        "I cannot",
        "I don't",
        "I'm not able",
        "I can't",
        "unclear",
        "vague",
        "depends",
        "it varies",
        "sorry",
        "apologize",
        "unfortunately",
    ]

    matches = []
    total = 0

    with open(tf_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                response = item["response"].lower()
                total += 1

                matched_patterns = []
                for pattern in unhelpful_patterns:
                    if pattern in response:
                        matched_patterns.append(pattern)

                if matched_patterns:
                    matches.append(
                        {
                            "prompt_id": item["prompt_id"],
                            "patterns": matched_patterns,
                            "response_preview": item["response"][:100],
                        }
                    )
            except:
                continue

    console.print(f"\n[bold]Unhelpful Pattern Analysis:[/bold]")
    console.print(f"Total responses: {total}")
    console.print(
        f"Responses with 'unhelpful' patterns: {len(matches)} ({len(matches)/total*100:.1f}%)"
    )

    # Show examples
    console.print("\n[yellow]Examples with unhelpful patterns:[/yellow]")
    for match in matches[:5]:
        console.print(f"\nID: {match['prompt_id']}")
        console.print(f"Patterns: {', '.join(match['patterns'])}")
        console.print(f"Response: {match['response_preview']}...")


def main():
    """Run the analysis."""
    console.print(
        "[bold blue]🔍 Analyzing Why pi_bad Still Has High SNIPS[/bold blue]\n"
    )

    # Check response alignment
    analyze_response_alignment()

    # Analyze policy preferences
    analyze_policy_preferences()

    # Check for unhelpful patterns
    check_unhelpful_alignment()

    console.print("\n[bold]🤔 Hypotheses:[/bold]")
    console.print(
        "1. **SNIPS is self-normalized**: It divides by sum of weights, reducing extreme weight impact"
    )
    console.print(
        "2. **Response style mismatch**: P0 (llama-8b) might produce responses that accidentally align with 'unhelpful' patterns"
    )
    console.print(
        "3. **Judge bias**: The judge might prefer certain response styles that correlate with pi_bad"
    )
    console.print(
        "4. **Temperature effect**: pi_bad uses temp=1.0 (more diverse) vs 0.5 for others"
    )


if __name__ == "__main__":
    main()
