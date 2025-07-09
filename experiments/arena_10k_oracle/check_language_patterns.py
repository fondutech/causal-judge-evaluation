#!/usr/bin/env python3
"""
Check language patterns in log probability failures.
"""

import json
from pathlib import Path
from collections import defaultdict
from rich.console import Console

console = Console()


def detect_language_simple(text):
    """Simple language detection based on common patterns."""
    # Check for common non-English characters/words
    if any(word in text.lower() for word in ['bonjour', 'merci', 'avec', 'pour', 'dans']):
        return 'fr'  # French
    elif any(word in text.lower() for word in ['hola', 'gracias', 'para', 'con', 'que']):
        return 'es'  # Spanish
    elif any(word in text.lower() for word in ['danke', 'bitte', 'mit', 'für', 'und']):
        return 'de'  # German
    elif any(char in text for char in 'абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'):
        return 'ru'  # Russian
    elif any(word in text.lower() for word in ['你好', '谢谢', '的', '是', '在']):
        return 'zh'  # Chinese
    else:
        return 'en'  # Default to English


def main():
    # Load all responses to check language distribution
    with open("phase1_dataset_preparation/data/all_responses.jsonl") as f:
        responses_data = [json.loads(line) for line in f]
    
    # Load log probs to check failures
    with open("phase1_dataset_preparation/data/logprobs.jsonl") as f:
        logprobs_data = {item["prompt_id"]: item for item in (json.loads(line) for line in f)}
    
    # Analyze language patterns
    language_stats = defaultdict(lambda: {"total": 0, "failed": 0, "zero_logp": 0})
    failed_samples = []
    
    for item in responses_data:
        prompt_id = item["prompt_id"]
        prompt_text = item["prompt"]
        
        # Detect language
        lang = detect_language_simple(prompt_text)
        language_stats[lang]["total"] += 1
        
        # Check if log probs failed
        if prompt_id in logprobs_data:
            logprob_item = logprobs_data[prompt_id]
            
            # Check for failures
            p0_logp = logprob_item.get("logprobs", {}).get("p0")
            if p0_logp is None:
                language_stats[lang]["failed"] += 1
                failed_samples.append((prompt_id, lang, prompt_text))
            elif p0_logp == 0.0:
                language_stats[lang]["zero_logp"] += 1
    
    # Display statistics
    console.print("\n[bold]Language Statistics:[/bold]")
    for lang, stats in sorted(language_stats.items()):
        failure_rate = stats["failed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        console.print(f"{lang}: {stats['total']} total, {stats['failed']} failed ({failure_rate:.1f}%), {stats['zero_logp']} zero")
    
    # Check specific failed examples
    console.print("\n[bold]Failed Sample Analysis:[/bold]")
    failed_ids = ["arena_sampled_11", "arena_sampled_18", "arena_sampled_24", "arena_sampled_46"]
    
    for prompt_id in failed_ids:
        for item in responses_data:
            if item["prompt_id"] == prompt_id:
                response = item["responses"]["p0"]["response"]
                console.print(f"\n{prompt_id}:")
                console.print(f"  Response length: {len(response)} chars")
                console.print(f"  Response preview: {response[:100]}...")
                break


    # Show failed samples
    console.print("\n[bold]Failed Samples:[/bold]")
    for prompt_id, lang, prompt_text in failed_samples:
        console.print(f"[red]{prompt_id}[/red] ({lang}): {prompt_text[:80]}...")


if __name__ == "__main__":
    main()