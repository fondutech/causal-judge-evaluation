#!/usr/bin/env python3
"""
Configuration loader for Arena 10K experiment.

Loads settings from the YAML config file to ensure consistency across all scripts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class ArenaConfig:
    """Load and access Arena 10K experiment configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader.

        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            # Default to the standard config location
            config_path = Path(__file__).parent.parent / "configs" / "arena_10k.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    @property
    def experiment_name(self) -> str:
        """Get experiment name."""
        return self.config["experiment"]["name"]

    @property
    def seed(self) -> int:
        """Get random seed."""
        return self.config["experiment"]["seed"]

    @property
    def work_dir(self) -> str:
        """Get working directory."""
        return self.config["paths"]["work_dir"]

    @property
    def logging_policy(self) -> Dict[str, Any]:
        """Get logging policy (P0) configuration."""
        return self.config["logging_policy"]

    @property
    def target_policies(self) -> List[Dict[str, Any]]:
        """Get list of target policy configurations."""
        return self.config["target_policies"]

    def get_target_policy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific target policy by name.

        Args:
            name: Policy name (e.g., 'pi_clone', 'pi_cot')

        Returns:
            Policy config dict or None if not found
        """
        for policy in self.target_policies:
            if policy["name"] == name:
                return policy
        return None

    @property
    def all_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get all policies (P0 + targets) as a dict keyed by name."""
        policies = {"p0": self.logging_policy}
        for target in self.target_policies:
            policies[target["name"]] = target
        return policies

    @property
    def judge_config(self) -> Dict[str, Any]:
        """Get judge configuration."""
        return self.config["judge"]

    @property
    def estimator_config(self) -> Dict[str, Any]:
        """Get estimator configuration."""
        return self.config["estimator"]

    def get_policy_model_config(self, policy_name: str) -> Dict[str, Any]:
        """Get model configuration for a specific policy.

        Args:
            policy_name: Name of policy ('p0', 'pi_clone', etc.)

        Returns:
            Dict with provider, model_name, temperature, max_tokens, etc.
        """
        if policy_name == "p0":
            return self.logging_policy

        policy = self.get_target_policy(policy_name)
        if policy is None:
            raise ValueError(f"Unknown policy: {policy_name}")

        # Remove the 'name' field as it's not a model parameter
        config = policy.copy()
        config.pop("name", None)
        return config

    def get_policy_system_prompt(self, policy_name: str) -> Optional[str]:
        """Get system prompt for a policy, handling suffixes.

        Args:
            policy_name: Name of policy

        Returns:
            Complete system prompt or None
        """
        config = self.get_policy_model_config(policy_name)

        # Handle explicit system prompt
        if "system_prompt" in config:
            return config["system_prompt"]

        # Handle system prompt suffix (for pi_cot)
        if "system_prompt_suffix" in config:
            base_prompt = "You are a helpful assistant."
            return base_prompt + config["system_prompt_suffix"]

        return None

    def get_policy_user_template(self, policy_name: str) -> str:
        """Get user message template for a policy.

        Args:
            policy_name: Name of policy

        Returns:
            User message template with {context} placeholder
        """
        # Special handling for pi_cot based on current implementation
        if policy_name == "pi_cot":
            return "{context}\n\nLet's think step by step."

        # Default template
        return "{context}"

    @property
    def uses_llama_cpp(self) -> bool:
        """Check if configuration uses llama.cpp instead of API providers."""
        # Check if any policy has type "llama_cpp"
        if self.logging_policy.get("type") == "llama_cpp":
            return True
        for policy in self.target_policies:
            if policy.get("type") == "llama_cpp":
                return True
        return False

    @property
    def llama_model_config(self) -> Optional[Dict[str, Any]]:
        """Get llama.cpp model configuration if present."""
        return self.config.get("model")


# Convenience function for scripts
def load_arena_config(config_path: Optional[str] = None) -> ArenaConfig:
    """Load Arena 10K configuration.

    Args:
        config_path: Optional path to config file

    Returns:
        ArenaConfig instance
    """
    return ArenaConfig(config_path)
