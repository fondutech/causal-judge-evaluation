"""Base provider strategy interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import os


class ProviderStrategy(ABC):
    """Abstract base class for API provider strategies."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the provider strategy.

        Args:
            api_key: API key for the provider (if None, will try environment variable)
            base_url: Base URL for the API (provider-specific)
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[Any] = None

    @abstractmethod
    def setup_client(self) -> Any:
        """Setup and return the API client for this provider.

        Returns:
            Configured API client instance

        Raises:
            ValueError: If required configuration is missing
        """
        pass

    @abstractmethod
    async def score(
        self,
        client: Any,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Score using this provider's API.

        Args:
            client: The API client instance
            prompt: The rendered prompt to send
            model_name: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Raw response content from the API

        Raises:
            Exception: Provider-specific API errors
        """
        pass

    @abstractmethod
    def get_default_env_var(self) -> str:
        """Get the default environment variable name for this provider's API key.

        Returns:
            Environment variable name (e.g., "OPENAI_API_KEY")
        """
        pass

    def get_api_key(self) -> str:
        """Get the API key, trying the provided key first, then environment variable.

        Returns:
            API key string

        Raises:
            ValueError: If no API key is found
        """
        api_key = self.api_key or os.getenv(self.get_default_env_var())
        if not api_key:
            raise ValueError(
                f"API key not found. Set {self.get_default_env_var()} environment variable "
                f"or provide api_key parameter."
            )
        return api_key

    def get_client(self) -> Any:
        """Get or create the API client instance.

        Returns:
            API client instance
        """
        if self._client is None:
            self._client = self.setup_client()
        return self._client
