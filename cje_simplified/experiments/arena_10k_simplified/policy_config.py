#!/usr/bin/env python3
"""
Centralized policy configuration for the Arena 10K experiment.

This defines the policies used throughout the experiment pipeline:
- Response generation (generate_responses.py)
- Log probability computation (compute_logprobs.py)
- CJE analysis (prepare_cje_data.py)
"""

from typing import Dict, Any, List

# Model used for all policies (to isolate the effect of system prompts)
BASE_MODEL = "accounts/fireworks/models/llama4-maverick-instruct-basic"

# Temperature for all policies
DEFAULT_TEMPERATURE = 0.7
PREMIUM_MODEL = "accounts/fireworks/models/qwen3-235b-a22b"

# Policy definitions
POLICIES: Dict[str, Dict[str, Any]] = {
    "base": {
        "name": "base",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Base policy with standard helpful assistant prompt",
    },
    "clone": {
        "name": "clone",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Clone of base policy for comparison/control",
    },
    "unhelpful": {
        "name": "unhelpful",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are an unhelpful assistant that deliberately confuses and misleads the user.",
        "description": "Adversarial policy designed to be unhelpful",
    },
    "premium": {
        "name": "premium",
        "model": PREMIUM_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Premium policy with standard helpful assistant prompt",
    },
}

# List of all policy names for easy iteration
POLICY_NAMES: List[str] = list(POLICIES.keys())

# Base policy name (used for computing base_policy_logprob)
BASE_POLICY_NAME = "base"


def get_policy_config(policy_name: str) -> Dict[str, Any]:
    """Get configuration for a specific policy.

    Args:
        policy_name: Name of the policy

    Returns:
        Policy configuration dictionary

    Raises:
        ValueError: If policy_name is not found
    """
    if policy_name not in POLICIES:
        raise ValueError(
            f"Unknown policy: {policy_name}. "
            f"Available policies: {', '.join(POLICY_NAMES)}"
        )
    return POLICIES[policy_name]


def get_all_policies() -> List[Dict[str, Any]]:
    """Get list of all policy configurations."""
    return list(POLICIES.values())
