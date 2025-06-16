#!/usr/bin/env python3
"""
Test the simplified completions template system (Llama 4 only).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List

from cje.loggers.completions_templates import (
    get_completions_template_for_model,
    list_available_completions_templates,
    CompletionsTemplate,
    Llama4CompletionsTemplate,
)


def test_llama4_template() -> None:
    """Test the Llama 4 template functionality."""
    print("=== Testing Llama 4 Template ===\n")

    # Test direct instantiation
    template = Llama4CompletionsTemplate()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    response = "The answer is 4."

    # Test formatting with response
    with_response = template.format_with_response(messages, response)
    print("With response:")
    print(repr(with_response))
    print(f"\nLength: {len(with_response)} chars")

    # Test formatting without response
    without_response = template.format_without_response(messages)
    print("\nWithout response:")
    print(repr(without_response))
    print(f"\nLength: {len(without_response)} chars")

    # Test EOS token
    print(f"\nEOS token: {repr(template.get_eos_token())}")

    # Verify correct format
    expected_start = "<|begin_of_text|><|header_start|>system<|header_end|>"
    assert with_response.startswith(
        expected_start
    ), f"Should start with {expected_start}"
    assert "<|eot|>" in with_response, "Should contain EOS tokens"
    assert response in with_response, f"Should contain response '{response}'"


def test_auto_detection() -> None:
    """Test that all models default to Llama 4 template."""
    print("\n=== Testing Auto-Detection (All Default to Llama 4) ===\n")

    test_cases = [
        ("fireworks", "accounts/fireworks/models/llama4-maverick-instruct"),
        ("fireworks", "accounts/fireworks/models/llama-4-scout"),
        ("fireworks", "any-other-model"),  # Still defaults to llama4
        # Other providers not yet supported for teacher forcing
    ]

    for provider, model_name in test_cases:
        template = get_completions_template_for_model(model_name, provider)
        print(f"{provider}:{model_name} -> {template.__class__.__name__}")
        assert isinstance(
            template, Llama4CompletionsTemplate
        ), f"All models should use Llama4CompletionsTemplate, got {type(template)}"


def test_template_registry() -> None:
    """Test that only llama4 template is available."""
    print("\n=== Testing Template Registry ===\n")

    available = list_available_completions_templates()
    print(f"Available templates: {available}")
    assert available == ["llama4"], f"Should only have llama4 template, got {available}"


def test_custom_template() -> None:
    """Test custom template implementation."""
    print("\n=== Testing Custom Template ===\n")

    class SimpleTemplate(CompletionsTemplate):
        def format_with_response(
            self, messages: List[Dict[str, str]], response: str
        ) -> str:
            return f"Messages: {messages} | Response: {response}"

        def format_without_response(self, messages: List[Dict[str, str]]) -> str:
            return f"Messages: {messages} | Response: "

        def get_eos_token(self) -> str:
            return "<END>"

    template = SimpleTemplate()
    messages = [{"role": "user", "content": "Hello"}]

    result = template.format_with_response(messages, "Hi")
    print(f"Custom format: {result}")
    assert "Hello" in result and "Hi" in result


def main() -> None:
    """Run all tests."""
    test_llama4_template()
    test_auto_detection()
    test_template_registry()
    test_custom_template()

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
