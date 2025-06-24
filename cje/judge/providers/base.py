"""Unified base provider strategy interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, cast
import os

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from ..schemas import (
    JudgeScore,
    JudgeEvaluation,
    DetailedJudgeEvaluation,
    JudgeScoreWithCI,
)

T = TypeVar("T", bound=BaseModel)


class UnifiedProviderStrategy(ABC):
    """Unified base class for API provider strategies with structured output support."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the provider strategy.

        Args:
            api_key: API key for the provider (if None, will try environment variable)
            base_url: Base URL for the API (provider-specific)
        """
        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[Any] = None

    # ------------------------------------------------------------------------
    # Legacy API (for non-structured usage)
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Structured Output API
    # ------------------------------------------------------------------------

    @abstractmethod
    def get_chat_model(
        self, model_name: str, temperature: float = 0.0
    ) -> BaseChatModel:
        """Get the LangChain chat model for this provider."""
        pass

    def get_structured_model(
        self,
        model_name: str,
        schema: Type[BaseModel],
        method: str = "auto",
        temperature: float = 0.0,
    ) -> Runnable[Any, Any]:
        """Get a model configured for structured output."""
        base_model = self.get_chat_model(model_name, temperature)

        # Determine the best method for this provider
        if method == "auto":
            method = self.get_recommended_method()

        # Configure structured output
        return self._configure_structured_output(base_model, schema, method)

    @abstractmethod
    def get_recommended_method(self) -> str:
        """Get the recommended structured output method for this provider."""
        pass

    def _configure_structured_output(
        self, model: BaseChatModel, schema: Type[BaseModel], method: str
    ) -> Runnable[Any, Any]:
        """Configure the model for structured output."""
        # Provider-specific parameters
        extra_params = self.get_structured_output_params(method)

        try:
            # The with_structured_output method returns a Runnable that outputs dict or BaseModel
            result = model.with_structured_output(
                schema, method=method, include_raw=True, **extra_params
            )
            return cast(Runnable[Any, Any], result)
        except Exception:
            # Fallback to basic configuration if provider doesn't support extra params
            result = model.with_structured_output(
                schema, method=method, include_raw=True
            )
            return cast(Runnable[Any, Any], result)

    def get_structured_output_params(self, method: str) -> Dict[str, Any]:
        """Get provider-specific parameters for structured output.

        Override in subclasses to add provider-specific parameters.
        """
        return {}

    def get_schema_class(self, schema_name: str) -> Type[BaseModel]:
        """Get the schema class by name."""
        schemas: Dict[str, Type[BaseModel]] = {
            "JudgeScore": JudgeScore,
            "JudgeEvaluation": JudgeEvaluation,
            "DetailedJudgeEvaluation": DetailedJudgeEvaluation,
            "JudgeScoreWithCI": JudgeScoreWithCI,
        }
        if schema_name not in schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        return schemas[schema_name]
