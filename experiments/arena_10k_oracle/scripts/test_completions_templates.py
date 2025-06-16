#!/usr/bin/env python3
"""
Test the completions template system.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from cje.loggers.completions_templates import (
    get_completions_template_for_model,
    list_available_completions_templates,
    CompletionsTemplateConfig,
    CompletionsTemplate,
)
from cje.loggers.api_policy import APIPolicyRunner


def test_template_detection() -> None:
    """Test automatic template detection based on model names."""
    print("=== Testing Template Auto-Detection ===\n")

    test_cases = [
        ("fireworks", "accounts/fireworks/models/llama-v3p1-70b-instruct", "llama3"),
        ("fireworks", "accounts/fireworks/models/llama4-maverick-instruct", "llama4"),
        ("fireworks", "accounts/fireworks/models/llama-4-scout", "llama4"),
        ("openai", "gpt-3.5-turbo", "chatml"),
        ("together", "some-alpaca-model", "alpaca"),
        ("anthropic", "claude-3-opus-20240229", "llama3"),
    ]

    for provider, model_name, expected in test_cases:
        template = get_completions_template_for_model(model_name, provider)
        actual = template.__class__.__name__
        # Handle special cases for capitalization
        if expected == "chatml":
            expected_class = "ChatMLCompletionsTemplate"
        else:
            expected_class = f"{expected.capitalize()}CompletionsTemplate"

        status = "✓" if expected_class in actual else "✗"
        print(f"{status} {provider}:{model_name} -> {actual}")

        if status == "✗":
            print(f"  Expected: {expected_class}")


def test_template_formatting() -> None:
    """Test template formatting with example messages."""
    print("\n=== Testing Template Formatting ===\n")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    response = "The answer is 4."

    templates = list_available_completions_templates()

    for template_name in templates:
        template = get_completions_template_for_model(
            "test-model", template_format=template_name
        )
        print(f"\n--- {template_name} ---")

        # Format with response
        with_response = template.format_with_response(messages, response)
        print(f"With response ({len(with_response)} chars):")
        print(
            repr(with_response[:100]) + "..."
            if len(with_response) > 100
            else repr(with_response)
        )

        # Format without response
        without_response = template.format_without_response(messages)
        print(f"\nWithout response ({len(without_response)} chars):")
        print(
            repr(without_response[:100]) + "..."
            if len(without_response) > 100
            else repr(without_response)
        )

        # EOS token
        print(f"\nEOS token: {repr(template.get_eos_token())}")


def test_api_policy_runner_integration() -> None:
    """Test APIPolicyRunner with the new template system."""
    print("\n=== Testing APIPolicyRunner Integration ===\n")

    try:
        # Test 1: Auto-detection
        runner = APIPolicyRunner(
            provider="fireworks",
            model_name="accounts/fireworks/models/llama4-maverick-instruct",
        )
        print(f"Auto-detected template: {runner.template.__class__.__name__}")

        # Test 2: Explicit format
        runner = APIPolicyRunner(
            provider="together", model_name="some-model", template_format="chatml"
        )
        print(f"Explicit template: {runner.template.__class__.__name__}")

    except Exception as e:
        print(
            f"Note: APIPolicyRunner initialization requires API keys: {type(e).__name__}"
        )
        print("Testing template system directly instead...\n")

    # Test 3: Direct template usage (doesn't require API keys)
    print("Testing custom template:")

    class MyCustomTemplate(CompletionsTemplate):
        def format_with_response(
            self, messages: List[Dict[str, str]], response: str
        ) -> str:
            return f"CUSTOM: {messages} -> {response}"

        def format_without_response(self, messages: List[Dict[str, str]]) -> str:
            return f"CUSTOM: {messages}"

        def get_eos_token(self) -> str:
            return "<END>"

    custom = MyCustomTemplate()
    messages = [{"role": "user", "content": "Hello"}]

    formatted_with = custom.format_with_response(messages, "Hi there")
    print(f"Custom format with response: {formatted_with}")

    formatted_without = custom.format_without_response(messages)
    print(f"Custom format without response: {formatted_without}")

    eos = custom.get_eos_token()
    print(f"Custom EOS token: {eos}")


def main() -> None:
    """Run all tests."""
    test_template_detection()
    test_template_formatting()
    test_api_policy_runner_integration()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
