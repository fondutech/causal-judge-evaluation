"""Tests for teacher forcing module.

Note: These tests focus on the template and conversion logic.
API integration tests would require actual credentials and are not included.
"""

import pytest
from typing import List, Dict, Any

from cje.teacher_forcing import (
    convert_chat_to_completions,
)


class TestChatConversion:
    """Test chat format conversion utilities."""

    def test_convert_chat_to_completions_basic(self) -> None:
        """Test basic chat to completion format conversion."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ]

        # Template config is required
        from cje.teacher_forcing.templates import ChatTemplateConfig

        # Create a minimal template config for testing
        class MinimalTemplate(ChatTemplateConfig):
            def format_message(self, role: str, content: str) -> str:
                return f"{role}: {content}"

            def format_message_header(self, role: str) -> str:
                return f"[{role}]"

            def should_add_bos(self) -> bool:
                return True

            begin_of_text = "<bos>"

        template = MinimalTemplate()
        prompt_only, prompt_plus_reply = convert_chat_to_completions(messages, template)

        # prompt_only should include everything except last assistant reply
        assert "Hello" in prompt_only
        assert "Hi there!" in prompt_only
        assert "How are you?" in prompt_only
        assert prompt_only.endswith("[assistant]")

        # prompt_plus_reply should include everything including last assistant reply
        assert "I'm doing well, thanks!" in prompt_plus_reply
        assert "Hello" in prompt_plus_reply

    def test_convert_chat_to_completions_with_system(self) -> None:
        """Test chat conversion with system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        from cje.teacher_forcing.templates import ChatTemplateConfig

        class MinimalTemplate(ChatTemplateConfig):
            def format_message(self, role: str, content: str) -> str:
                return f"{role}: {content}"

            def format_message_header(self, role: str) -> str:
                return f"[{role}]"

            def should_add_bos(self) -> bool:
                return True

            begin_of_text = "<bos>"

        template = MinimalTemplate()
        prompt_only, prompt_plus_reply = convert_chat_to_completions(messages, template)

        # System message should be in prompt_only
        assert "helpful assistant" in prompt_only
        assert "2+2" in prompt_only
        assert prompt_only.endswith("[assistant]")

        # prompt_plus_reply should include the answer
        assert "assistant: 4" in prompt_plus_reply

    def test_convert_chat_empty_messages(self) -> None:
        """Test chat conversion with empty messages."""
        from cje.teacher_forcing.templates import ChatTemplateConfig

        class MinimalTemplate(ChatTemplateConfig):
            def format_message(self, role: str, content: str) -> str:
                return f"{role}: {content}"

            def format_message_header(self, role: str) -> str:
                return f"[{role}]"

            def should_add_bos(self) -> bool:
                return True

            begin_of_text = "<bos>"

        template = MinimalTemplate()

        with pytest.raises(ValueError, match="Chat cannot be empty"):
            convert_chat_to_completions([], template)

    def test_convert_chat_no_assistant(self) -> None:
        """Test chat conversion without assistant message."""
        messages = [{"role": "user", "content": "Hello"}]

        from cje.teacher_forcing.templates import ChatTemplateConfig

        class MinimalTemplate(ChatTemplateConfig):
            def format_message(self, role: str, content: str) -> str:
                return f"{role}: {content}"

            def format_message_header(self, role: str) -> str:
                return f"[{role}]"

            def should_add_bos(self) -> bool:
                return True

            begin_of_text = "<bos>"

        template = MinimalTemplate()

        with pytest.raises(ValueError, match="Last message must be assistant reply"):
            convert_chat_to_completions(messages, template)

    def test_convert_chat_wrong_last_role(self) -> None:
        """Test chat conversion with non-assistant last message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]

        from cje.teacher_forcing.templates import ChatTemplateConfig

        class MinimalTemplate(ChatTemplateConfig):
            def format_message(self, role: str, content: str) -> str:
                return f"{role}: {content}"

            def format_message_header(self, role: str) -> str:
                return f"[{role}]"

            def should_add_bos(self) -> bool:
                return True

            begin_of_text = "<bos>"

        template = MinimalTemplate()

        with pytest.raises(ValueError, match="Last message must be assistant reply"):
            convert_chat_to_completions(messages, template)


# Note: API integration tests would go here but require actual API keys
# and should be marked with @pytest.mark.integration and @pytest.mark.slow
# They should be skipped by default and only run with specific flags
