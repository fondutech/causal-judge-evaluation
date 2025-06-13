# mypy: ignore-errors
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict
import logging
import openai  # For error types in OpenAIAdapter

# Import error types for Anthropic and Gemini when their actual calls are implemented
# e.g., from anthropic import RateLimitError as AnthropicRateLimitError, APIError as AnthropicAPIError
# e.g., from google.api_core.exceptions import GoogleAPICallError, RetryError as GoogleRetryError

from cje.utils.retry import retry_api_call  # C-5: Import retry utility
from cje.loggers.conversation_utils import parse_context
from cje.utils.progress import track

# Real SDK imports for default client instantiation
import anthropic
import google.generativeai as genai
import os  # For Google API key

logger = logging.getLogger(__name__)

# Public re-exports for easy importing elsewhere
__all__ = [
    "ProviderAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OpenAICompatibleAdapter",
    "FireworksAdapter",
    "TogetherAdapter",
]


class ProviderAdapter(ABC):
    """Abstract base class for provider-specific API request logic."""

    def __init__(
        self,
        model_name: str,
        client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
    ):
        """
        Initializes the adapter with a model name and an optional client instance.
        If client is None, a default real SDK client will be instantiated.

        Args:
            model_name: The name of the model to be used.
            client: An optional pre-initialized API client for the provider.
            system_prompt: Optional system prompt to prepend to conversations.
            user_message_template: Template for formatting user messages.
        """
        self.model_name = model_name
        self.client = client
        self.system_prompt = system_prompt
        self.user_message_template = user_message_template
        # Specific client instantiation will be handled in subclass __init__ if self.client is None

    def _parse_context(self, context: str) -> List[Dict[str, str]]:
        """Parse context into proper message format using shared utilities."""
        return parse_context(context, self.system_prompt, self.user_message_template)

    @abstractmethod
    def request(
        self, prompts: List[str], progress: bool = False, **kwargs: Any
    ) -> List[Tuple[str, List[float]]]:
        """
        Sends requests to the provider's API and returns generated text with token logprobs.

        Args:
            prompts: A list of prompt strings.
            progress: If True, display a progress bar over chunks.
            **kwargs: Additional keyword arguments for the API request, including batch_size.

        Returns:
            A list of tuples, where each tuple contains the generated text (str)
            and a list of token logprobs (List[float]).
        """
        pass


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API."""

    def __init__(
        self,
        model_name: str,
        client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
    ):
        super().__init__(model_name, client, system_prompt, user_message_template)
        # Initialize the client if None
        if self.client is None:
            try:
                import openai

                self.client = openai.OpenAI()
            except Exception as e:
                raise ImportError(
                    f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY is set."
                ) from e

        # At this point, self.client should never be None
        assert self.client is not None, "OpenAI client is None after initialization"

        # Verify client has required attributes
        if not hasattr(self.client, "chat") or not hasattr(
            self.client.chat, "completions"
        ):
            raise ValueError(
                f"OpenAI client missing required attributes: 'chat.completions'. Got client type: {type(self.client)}"
            )

    def request(
        self, prompts: List[str], progress: bool = False, **kwargs: Any
    ) -> List[Tuple[str, List[float]]]:
        """
        Sends requests to the OpenAI API for each prompt and returns generated text with token logprobs.
        Handles batching of prompts if the total number exceeds batch_size.

        Args:
            prompts: A list of prompt strings.
            progress: If True, display a progress bar over chunks.
            **kwargs: Additional keyword arguments for the OpenAI API chat completion request
                      (e.g., max_new_tokens, temperature, top_p, batch_size).

        Returns:
            A list of tuples, where each tuple contains the generated text (str)
            and a list of its token logprobs (List[float]).
        """
        batch_size = kwargs.get("batch_size", 16)
        all_results: List[Tuple[str, List[float]]] = []

        api_request_base_kwargs = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 1.0),
            "max_tokens": kwargs.get("max_new_tokens", 128),
            "logprobs": True,
        }

        num_prompts = len(prompts)
        # Create chunks for progress tracking
        chunks = []
        for i in range(0, num_prompts, batch_size):
            chunks.append(prompts[i : i + batch_size])

        # Process chunks with progress
        for chunk_prompts in track(
            chunks,
            description=f"OpenAI Processing ({batch_size} per batch)",
            disable=not progress,
            total=len(chunks),
        ):
            for prompt in chunk_prompts:
                # Parse the prompt into proper message format
                messages = self._parse_context(prompt)

                try:
                    # C-5: Wrap API call with retry_api_call
                    response = retry_api_call(
                        self.client.chat.completions.create,
                        messages=messages,
                        **api_request_base_kwargs,
                    )

                    completion_text: str = ""
                    token_logprobs_list: List[float] = []

                    if (
                        response and response.choices
                    ):  # Ensure response and choices exist
                        choice = response.choices[0]
                        completion_text = (
                            choice.message.content
                            if choice.message and choice.message.content
                            else ""
                        )

                        if choice.logprobs and choice.logprobs.content:
                            for token_info in choice.logprobs.content:
                                token_logprobs_list.append(
                                    token_info.logprob
                                    if token_info.logprob is not None
                                    else 0.0
                                )

                    all_results.append((completion_text, token_logprobs_list))

                except Exception as e:
                    logger.error(
                        f"Error in OpenAI API call: {e}. "
                        f"Prompt: {prompt[:100]}... "
                        f"Model: {self.model_name}"
                    )
                    raise

        return all_results


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API."""

    def __init__(
        self,
        model_name: str,
        client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
    ):
        super().__init__(model_name, client, system_prompt, user_message_template)
        if self.client is None:
            try:
                import anthropic

                self.client = anthropic.Anthropic()
            except Exception as e:
                raise ImportError(
                    f"Failed to initialize Anthropic client: {e}. Ensure ANTHROPIC_API_KEY is set."
                ) from e

        # At this point, self.client should never be None
        assert self.client is not None, "Anthropic client is None after initialization"

        # Verify client has required attributes
        if not hasattr(self.client, "messages"):
            raise ValueError(
                f"Anthropic client missing required attribute: 'messages'. Got client type: {type(self.client)}"
            )

    def request(
        self, prompts: List[str], progress: bool = False, **kwargs: Any
    ) -> List[Tuple[str, List[float]]]:
        """
        Sends requests to the Anthropic API and returns generated text with token logprobs.
        Handles batching of prompts if the total number exceeds batch_size.

        Args:
            prompts: A list of prompt strings.
            progress: If True, display a progress bar over chunks.
            **kwargs: Additional keyword arguments for the Anthropic API request (e.g., batch_size).

        Returns:
            A list of tuples, where each tuple contains the generated text (str)
            and a list of token logprobs (List[float]).
        """
        batch_size = kwargs.get("batch_size", 16)
        all_results: List[Tuple[str, List[float]]] = []

        num_prompts = len(prompts)
        # Create chunks for progress tracking
        chunks = []
        for i in range(0, num_prompts, batch_size):
            chunks.append(prompts[i : i + batch_size])

        # Process chunks with progress
        for chunk_prompts in track(
            chunks,
            description=f"Anthropic Processing ({batch_size} per batch)",
            disable=not progress,
            total=len(chunks),
        ):
            for prompt in chunk_prompts:
                # Parse the prompt into proper message format
                messages = self._parse_context(prompt)

                # Anthropic handles system messages differently
                system_content = None
                filtered_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_content = msg["content"]
                    else:
                        filtered_messages.append(msg)

                try:
                    api_kwargs = {
                        "model": self.model_name,
                        "messages": filtered_messages,
                        "max_tokens": kwargs.get("max_new_tokens", 128),
                        "temperature": kwargs.get("temperature", 0.0),
                        "top_p": kwargs.get("top_p", 1.0),
                        "logprobs": True,
                    }

                    if system_content:
                        api_kwargs["system"] = system_content

                    response = retry_api_call(self.client.messages.create, **api_kwargs)

                    completion_text = ""
                    token_logprobs_list: List[float] = []

                    if (
                        hasattr(response, "content")
                        and isinstance(response.content, list)
                        and response.content
                        and isinstance(response.content[0], dict)
                    ):
                        completion_text = response.content[0].get("text", "")

                    if hasattr(response, "token_logprobs") and isinstance(
                        response.token_logprobs, list
                    ):
                        token_logprobs_list = [
                            float(lp) for lp in response.token_logprobs
                        ]

                    all_results.append((completion_text, token_logprobs_list))
                except Exception as e:
                    logger.error(
                        f"Error in Anthropic API call: {e}. "
                        f"Prompt: {prompt[:100]}... "
                        f"Model: {self.model_name}"
                    )
                    raise

        return all_results


class GeminiAdapter(ProviderAdapter):
    """Adapter for Google Gemini API."""

    # Note: The Gemini client (GenerativeModel) is often model-specific itself.
    # The `client` passed to __init__ here would be the `genai.GenerativeModel` instance.
    def __init__(
        self,
        model_name: str,
        client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
    ):
        super().__init__(model_name, client, system_prompt, user_message_template)
        if self.client is None:
            try:
                import google.generativeai as genai

                if not os.getenv("GOOGLE_API_KEY"):
                    raise ValueError("GOOGLE_API_KEY environment variable not set.")
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.client = genai.GenerativeModel(
                    self.model_name
                )  # model_name is used here
            except Exception as e:
                raise ImportError(
                    f"Failed to initialize Google Gemini client for model {self.model_name}: {e}"
                ) from e

        # At this point, self.client should never be None
        assert self.client is not None, "Gemini client is None after initialization"

        # Verify client has required attributes
        if not hasattr(self.client, "generate_content"):
            raise ValueError(
                f"Gemini client missing required attribute: 'generate_content'. Got client type: {type(self.client)}"
            )

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert message array to text format for Gemini."""
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text_parts.append(f"System: {content}")
            elif role == "user":
                text_parts.append(f"Human: {content}")
            elif role == "assistant":
                text_parts.append(f"AI: {content}")
        return "\n".join(text_parts)

    def request(
        self, prompts: List[str], progress: bool = False, **kwargs: Any
    ) -> List[Tuple[str, List[float]]]:
        """
        Sends requests to the Google Gemini API and returns generated text with token logprobs.
        Handles batching of prompts if the total number exceeds batch_size.

        Args:
            prompts: A list of prompt strings.
            progress: If True, display a progress bar over chunks.
            **kwargs: Additional keyword arguments for the Google Gemini API request (e.g., batch_size).

        Returns:
            A list of tuples, where each tuple contains the generated text (str)
            and a list of token logprobs (List[float]).
        """
        batch_size = kwargs.get("batch_size", 16)
        all_results: List[Tuple[str, List[float]]] = []

        num_prompts = len(prompts)
        # Create chunks for progress tracking
        chunks = []
        for i in range(0, num_prompts, batch_size):
            chunks.append(prompts[i : i + batch_size])

        # Process chunks with progress
        for chunk_prompts in track(
            chunks,
            description=f"Gemini Processing ({batch_size} per batch)",
            disable=not progress,
            total=len(chunks),
        ):
            for prompt in chunk_prompts:
                # Parse the prompt into proper message format, then convert to text
                messages = self._parse_context(prompt)
                formatted_prompt = self._messages_to_text(messages)

                try:
                    response = retry_api_call(
                        self.client.generate_content,
                        formatted_prompt,
                        generation_config={
                            "temperature": kwargs.get("temperature", 0.0),
                            "top_p": kwargs.get("top_p", 1.0),
                            "max_output_tokens": kwargs.get("max_new_tokens", 128),
                        },
                    )

                    completion_text = getattr(response, "text", "")
                    token_logprobs_list: List[float] = []

                    if hasattr(response, "token_logprobs") and isinstance(
                        response.token_logprobs, list
                    ):
                        token_logprobs_list = [
                            float(lp) for lp in response.token_logprobs
                        ]

                    all_results.append((completion_text, token_logprobs_list))
                except Exception as e:
                    logger.error(
                        f"Error in Google API call: {e}. "
                        f"Prompt: {prompt[:100]}... "
                        f"Model: {self.model_name}"
                    )
                    raise

        return all_results


class OpenAICompatibleAdapter(OpenAIAdapter):
    """Generic adapter for OpenAI-style back-ends (base_url + API-key env var)."""

    ENV_VAR: str = "OPENAI_API_KEY"
    DEFAULT_BASE_URL: str | None = None

    def __init__(
        self,
        model_name: str,
        *,
        client: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
    ) -> None:
        if client is None:
            import openai, os

            client = openai.OpenAI(
                api_key=os.getenv(self.ENV_VAR),
                base_url=self.DEFAULT_BASE_URL or None,
            )
        super().__init__(
            model_name=model_name,
            client=client,
            system_prompt=system_prompt,
            user_message_template=user_message_template,
        )


class FireworksAdapter(OpenAICompatibleAdapter):
    ENV_VAR = "FIREWORKS_API_KEY"
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


class TogetherAdapter(OpenAICompatibleAdapter):
    ENV_VAR = "TOGETHER_API_KEY"
    DEFAULT_BASE_URL = "https://api.together.xyz/v1"
