"""HuggingFace tokenizer-based template configuration."""

from typing import List, Dict
from .base import ChatTemplateConfig


class HuggingFaceTemplateConfig(ChatTemplateConfig):
    """Auto-detect template from HuggingFace tokenizer.

    This uses the tokenizer's apply_chat_template method to handle
    formatting automatically.
    """

    def __init__(self, tokenizer_name: str):
        """Initialize with HuggingFace model/tokenizer name.

        Args:
            tokenizer_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        """
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Load the tokenizer."""
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        except ImportError:
            raise ImportError(
                "transformers library required for HuggingFace template support. "
                "Install with: pip install transformers"
            )

    def format_message(self, role: str, content: str) -> str:
        """Format a complete message using tokenizer."""
        # For single message formatting, we'll use the tokenizer with a single-message chat
        chat = [{"role": role, "content": content}]
        return self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )

    def format_message_header(self, role: str) -> str:
        """Format just the header using tokenizer."""
        # Create empty message and extract just the header part
        chat = [{"role": role, "content": ""}]
        full = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False
        )
        # Remove the trailing EOT/end token to get just the header
        # This is a bit hacky but works for most formats
        return full.rstrip().rstrip("<|eot_id|>").rstrip("<|eot|>").rstrip("<|im_end|>")

    def should_add_bos(self) -> bool:
        """HF tokenizer handles BOS automatically."""
        return False

    def apply_chat_template(
        self, chat: List[Dict[str, str]], add_generation_prompt: bool = False
    ) -> str:
        """Direct access to tokenizer's apply_chat_template for complex cases."""
        return self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )
