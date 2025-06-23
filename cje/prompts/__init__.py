"""
Unified Prompt Template System for CJE

This module provides a single, powerful templating system that handles
both judge prompts and LLM policy prompts with the same level of sophistication.

Everything is unified - no more separate systems!
"""

from .manager import (
    TemplateManager,
    render_policy,
    render_judge,
    list_templates,
    TemplateType,
)
from .unified_templates import UNIFIED_TEMPLATES, TEMPLATE_CATEGORIES
from .judge_templates import JUDGE_TEMPLATES
from .tuning import tune_judge_prompt

__all__ = [
    # Unified template system
    "TemplateManager",
    "render_policy",
    "render_judge",
    "list_templates",
    "TemplateType",
    "UNIFIED_TEMPLATES",
    "TEMPLATE_CATEGORIES",
    "JUDGE_TEMPLATES",
    # Tuning utilities
    "tune_judge_prompt",
]
