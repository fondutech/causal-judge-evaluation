#!/usr/bin/env python3
"""
Centralized policy configuration for the Arena 10K experiment.

This defines the policies used throughout the experiment pipeline:
- Response generation (generate_responses.py)
- Log probability computation (compute_logprobs.py)
- CJE analysis (prepare_cje_data.py)
"""

from typing import Dict, Any, List, Optional

# Model used for all policies (to isolate the effect of system prompts)
BASE_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
PREMIUM_MODEL = "accounts/fireworks/models/llama-v3p1-405b-instruct"

# Temperature for all policies
DEFAULT_TEMPERATURE = 0.7


# Policy definitions
POLICIES: Dict[str, Dict[str, Any]] = {
    "base": {
        "name": "base",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Base policy with standard helpful assistant prompt",
        "template_config": None,  # Will auto-detect
    },
    "clone": {
        "name": "clone",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Clone of base policy for comparison/control",
        "template_config": None,  # Will auto-detect
    },
    "unhelpful": {
        "name": "unhelpful",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are an unhelpful assistant that deliberately confuses and misleads the user.",
        "description": "Adversarial policy designed to be unhelpful",
        "template_config": None,  # Will auto-detect
    },
    "parallel_universe_prompt": {
        "name": "parallel_universe_prompt",
        "model": BASE_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "Imagine parallel universes where you vary your responses and can observe which one improves the user's life the most. Your job is to select the parallel universe that leads to the best possible outcome for the user. Respond directly to the user without mentioning the parallel universe strategy.",
        "description": "Parallel universe prompt",
        "template_config": None,  # Will auto-detect
    },
    "premium": {
        "name": "premium",
        "model": PREMIUM_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "system_prompt": "You are a helpful assistant.",
        "description": "Premium policy with Llama 405B model",
        "template_config": None,  # Will auto-detect
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
