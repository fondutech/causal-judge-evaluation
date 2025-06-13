"""AWS Secrets Manager utility for secure API key retrieval."""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from functools import lru_cache

try:
    import boto3
    from botocore.exceptions import (
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
    )

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretsManagerError(Exception):
    """Base exception for Secrets Manager operations."""

    pass


class SecretsManager:
    """AWS Secrets Manager client with caching and error handling."""

    def __init__(
        self, region_name: Optional[str] = None, profile_name: Optional[str] = None
    ):
        """Initialize the Secrets Manager client.

        Args:
            region_name: AWS region name (defaults to AWS_DEFAULT_REGION env var)
            profile_name: AWS profile name (optional)
        """
        if not BOTO3_AVAILABLE:
            raise SecretsManagerError(
                "boto3 is required for AWS Secrets Manager integration. "
                "Install with: pip install boto3"
            )

        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.profile_name = profile_name
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy initialization of boto3 client."""
        if self._client is None:
            try:
                session = boto3.Session(profile_name=self.profile_name)
                self._client = session.client(
                    "secretsmanager", region_name=self.region_name
                )
            except (NoCredentialsError, PartialCredentialsError) as e:
                raise SecretsManagerError(
                    f"AWS credentials not configured properly: {e}\n"
                    "Please configure credentials using AWS CLI, IAM roles, or environment variables."
                ) from e
        return self._client

    @lru_cache(maxsize=32)
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve and parse a secret from AWS Secrets Manager.

        Args:
            secret_name: Name or ARN of the secret

        Returns:
            Dictionary containing the secret data

        Raises:
            SecretsManagerError: If secret retrieval fails
        """
        try:
            logger.debug(f"Retrieving secret: {secret_name}")
            response = self.client.get_secret_value(SecretId=secret_name)

            # Parse the secret string (assuming JSON format)
            secret_string = response.get("SecretString")
            if not secret_string:
                raise SecretsManagerError(f"Secret {secret_name} has no SecretString")

            try:
                parsed_data: Dict[str, Any] = json.loads(secret_string)
                return parsed_data
            except json.JSONDecodeError:
                # If not JSON, return as a single key-value pair
                return {"value": secret_string}

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code == "ResourceNotFoundException":
                raise SecretsManagerError(f"Secret '{secret_name}' not found") from e
            elif error_code == "InvalidParameterException":
                raise SecretsManagerError(
                    f"Invalid parameter for secret '{secret_name}': {error_message}"
                ) from e
            elif error_code == "InvalidRequestException":
                raise SecretsManagerError(
                    f"Invalid request for secret '{secret_name}': {error_message}"
                ) from e
            elif error_code == "DecryptionFailureException":
                raise SecretsManagerError(
                    f"Failed to decrypt secret '{secret_name}'"
                ) from e
            else:
                raise SecretsManagerError(
                    f"Failed to retrieve secret '{secret_name}': {error_message}"
                ) from e

    def get_secret_value(self, secret_name: str, key: Optional[str] = None) -> str:
        """Get a specific value from a secret.

        Args:
            secret_name: Name or ARN of the secret
            key: Specific key within the secret (if secret is JSON with multiple keys)

        Returns:
            The secret value as a string
        """
        secret_data = self.get_secret(secret_name)

        if key is None:
            # If no key specified, try common patterns
            if "value" in secret_data:
                return str(secret_data["value"])
            elif len(secret_data) == 1:
                return str(next(iter(secret_data.values())))
            else:
                raise SecretsManagerError(
                    f"Secret '{secret_name}' contains multiple keys: {list(secret_data.keys())}. "
                    "Please specify which key to use."
                )
        else:
            if key not in secret_data:
                raise SecretsManagerError(
                    f"Key '{key}' not found in secret '{secret_name}'. "
                    f"Available keys: {list(secret_data.keys())}"
                )
            return str(secret_data[key])


# Global instance for convenience
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(
    region_name: Optional[str] = None, profile_name: Optional[str] = None
) -> SecretsManager:
    """Get or create a SecretsManager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(
            region_name=region_name, profile_name=profile_name
        )
    return _secrets_manager


def get_api_key_from_secrets(
    secret_name: str,
    key: Optional[str] = None,
    env_var_name: Optional[str] = None,
    cache_in_env: bool = True,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> str:
    """Retrieve an API key from AWS Secrets Manager.

    This is the main convenience function for getting API keys from Secrets Manager.

    Args:
        secret_name: Name or ARN of the secret in AWS Secrets Manager
        key: Specific key within the secret (if secret contains multiple keys)
        env_var_name: Environment variable name to cache the key in (optional)
        cache_in_env: Whether to cache the retrieved key in environment variable
        region_name: AWS region name (optional)
        profile_name: AWS profile name (optional)

    Returns:
        The API key as a string

    Examples:
        # Simple secret with single value
        api_key = get_api_key_from_secrets("openai-api-key")

        # JSON secret with multiple keys
        api_key = get_api_key_from_secrets("api-keys", key="openai")

        # Cache in environment variable
        api_key = get_api_key_from_secrets(
            "openai-api-key",
            env_var_name="OPENAI_API_KEY",
            cache_in_env=True
        )
    """
    # Check if already cached in environment
    if env_var_name and cache_in_env:
        cached_key = os.getenv(env_var_name)
        if cached_key:
            logger.debug(f"Using cached API key from {env_var_name}")
            return cached_key

    # Retrieve from Secrets Manager
    try:
        secrets_manager = get_secrets_manager(
            region_name=region_name, profile_name=profile_name
        )
        api_key = secrets_manager.get_secret_value(secret_name, key)

        # Cache in environment variable if requested
        if env_var_name and cache_in_env:
            os.environ[env_var_name] = api_key
            logger.debug(f"Cached API key in {env_var_name}")

        return api_key

    except SecretsManagerError as e:
        logger.error(f"Failed to retrieve API key from Secrets Manager: {e}")
        raise


def setup_api_keys_from_secrets(
    secrets_config: Dict[str, Union[str, Dict[str, Any]]],
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> None:
    """Set up multiple API keys from AWS Secrets Manager.

    Args:
        secrets_config: Configuration mapping environment variable names to secret configs
        region_name: AWS region name (optional)
        profile_name: AWS profile name (optional)

    Example:
        setup_api_keys_from_secrets({
            "OPENAI_API_KEY": "openai-api-key",  # Simple secret name
            "ANTHROPIC_API_KEY": {
                "secret_name": "anthropic-keys",
                "key": "api_key"
            }
        })
    """
    for env_var_name, config in secrets_config.items():
        try:
            if isinstance(config, str):
                # Simple secret name
                secret_name = config
                key = None
            else:
                # Dictionary config
                secret_name = config["secret_name"]
                key = config.get("key")

            api_key = get_api_key_from_secrets(
                secret_name=secret_name,
                key=key,
                env_var_name=env_var_name,
                cache_in_env=True,
                region_name=region_name,
                profile_name=profile_name,
            )

            logger.info(
                f"Successfully configured {env_var_name} from AWS Secrets Manager"
            )

        except SecretsManagerError as e:
            logger.warning(f"Failed to configure {env_var_name}: {e}")
            # Continue with other keys even if one fails


# Convenience functions for common API keys
def get_openai_api_key(
    secret_name: str = "openai-api-key",
    key: Optional[str] = None,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> str:
    """Get OpenAI API key from AWS Secrets Manager."""
    return get_api_key_from_secrets(
        secret_name=secret_name,
        key=key,
        env_var_name="OPENAI_API_KEY",
        cache_in_env=True,
        region_name=region_name,
        profile_name=profile_name,
    )


def get_anthropic_api_key(
    secret_name: str = "anthropic-api-key",
    key: Optional[str] = None,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> str:
    """Get Anthropic API key from AWS Secrets Manager."""
    return get_api_key_from_secrets(
        secret_name=secret_name,
        key=key,
        env_var_name="ANTHROPIC_API_KEY",
        cache_in_env=True,
        region_name=region_name,
        profile_name=profile_name,
    )


def get_google_api_key(
    secret_name: str = "google-api-key",
    key: Optional[str] = None,
    region_name: Optional[str] = None,
    profile_name: Optional[str] = None,
) -> str:
    """Get Google API key from AWS Secrets Manager."""
    return get_api_key_from_secrets(
        secret_name=secret_name,
        key=key,
        env_var_name="GOOGLE_API_KEY",
        cache_in_env=True,
        region_name=region_name,
        profile_name=profile_name,
    )
