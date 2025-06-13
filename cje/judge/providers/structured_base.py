"""Base provider with structured output support."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast
from pydantic import BaseModel

# Import LangChain components
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..schemas import JudgeScore, JudgeEvaluation, DetailedJudgeEvaluation

T = TypeVar("T", bound=BaseModel)


class StructuredProviderStrategy(ABC):
    """Abstract base for providers with structured output support."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

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
        }
        if schema_name not in schemas:
            raise ValueError(f"Unknown schema: {schema_name}")
        return schemas[schema_name]
