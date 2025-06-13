"""
Unified Template System for CJE

This module provides a single, powerful templating system that handles
both judge prompts and LLM policy prompts with the same level of sophistication.
"""

from typing import Dict, Any, List, Optional
import jinja2

# Unified template registry - handles both judges and policies
UNIFIED_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # === POLICY TEMPLATES ===
    "helpful_assistant": {
        "type": "policy",
        "system_prompt": "You are a helpful assistant. {{behavior_instructions}}",
        "user_template": "{{context}}",
        "variables": {
            "behavior_instructions": "Provide clear, accurate, and concise responses."
        },
        "description": "General-purpose helpful assistant",
        "category": "basic",
    },
    "chain_of_thought": {
        "type": "policy",
        "system_prompt": "You are an expert problem solver. {{reasoning_style}} {{output_format}}",
        "user_template": "Let's work through this step by step.\n\n{{context}}\n\n{{step_prompt}}",
        "variables": {
            "reasoning_style": "Think systematically and show your reasoning clearly.",
            "output_format": "Structure your response with numbered steps and clear conclusions.",
            "step_prompt": "Step 1:",
        },
        "description": "Step-by-step reasoning with explicit thought process",
        "category": "reasoning",
    },
    "domain_expert": {
        "type": "policy",
        "system_prompt": "You are a {{expertise_level}} {{domain}} expert. {{credentials}} {{communication_style}}",
        "user_template": "{{task_type}}: {{context}}\n\n{{instruction}}",
        "variables": {
            "expertise_level": "senior",
            "domain": "technical",
            "credentials": "You have extensive experience and deep knowledge in your field.",
            "communication_style": "Explain concepts clearly and provide actionable insights.",
            "task_type": "Question",
            "instruction": "Please provide your expert analysis:",
        },
        "description": "Configurable domain expert with customizable expertise",
        "category": "professional",
    },
    "conversational": {
        "type": "policy",
        "system_prompt": "You are a {{personality}} AI assistant. {{conversation_style}} {{tone_guidance}}",
        "user_template": "{{context}}",
        "variables": {
            "personality": "friendly and engaging",
            "conversation_style": "Be natural and maintain context throughout our conversation.",
            "tone_guidance": "Match your tone to the user's needs and the topic at hand.",
        },
        "description": "Natural conversational AI with personality",
        "category": "basic",
    },
    "few_shot_learner": {
        "type": "policy",
        "system_prompt": "You are an expert in {{domain}}. {{learning_instruction}}",
        "user_template": "Here are examples of excellent {{domain}} responses:\n\n{{examples}}\n\nNow handle this: {{context}}",
        "variables": {
            "domain": "question answering",
            "learning_instruction": "Follow the pattern and quality demonstrated in the examples.",
            "examples": "Q: What is gravity?\nA: Gravity is the fundamental force that attracts objects with mass toward each other.\n\nQ: What is photosynthesis?\nA: Photosynthesis is the process plants use to convert sunlight into chemical energy.",
        },
        "description": "Few-shot learning with customizable examples",
        "category": "learning",
    },
    # === JUDGE TEMPLATES ===
    "comprehensive_judge": {
        "type": "judge",
        "template": """Evaluate this response carefully:

CONTEXT: {{context}}

RESPONSE: {{response}}

Please rate this response on the following criteria:
{{criteria}}

Consider:
- Accuracy and factual correctness
- Relevance to the context
- Clarity and coherence
- Completeness of the answer
{{additional_considerations}}

Provide your rating as a number from {{min_score}} to {{max_score}}, where:
- {{min_score}}: {{min_description}}
- {{max_score}}: {{max_description}}

Rating: """,
        "variables": {
            "criteria": "• Accuracy\n• Relevance\n• Clarity\n• Completeness",
            "additional_considerations": "• Helpfulness to the user",
            "min_score": "1",
            "max_score": "10",
            "min_description": "Poor quality, incorrect or unhelpful",
            "max_description": "Excellent quality, accurate and very helpful",
        },
        "description": "Comprehensive multi-criteria evaluation",
        "category": "evaluation",
    },
    "reasoning_judge": {
        "type": "judge",
        "template": """Evaluate the reasoning quality in this response:

CONTEXT: {{context}}

RESPONSE: {{response}}

Assessment criteria:
{{reasoning_criteria}}

Rate the reasoning quality from {{min_score}} to {{max_score}}:
{{score_guide}}

Focus on: {{focus_areas}}

Score: """,
        "variables": {
            "reasoning_criteria": "• Logical flow and structure\n• Evidence and support\n• Clarity of explanation\n• Completeness of reasoning",
            "min_score": "1",
            "max_score": "5",
            "score_guide": "1=Poor reasoning, 2=Weak reasoning, 3=Adequate reasoning, 4=Good reasoning, 5=Excellent reasoning",
            "focus_areas": "logical consistency, evidence quality, and explanation clarity",
        },
        "description": "Focused evaluation of reasoning quality",
        "category": "evaluation",
    },
    "quick_judge": {
        "type": "judge",
        "template": """{{context}} -> {{response}}

{{evaluation_instruction}}

Score ({{score_range}}): """,
        "variables": {
            "evaluation_instruction": "Rate this response quality",
            "score_range": "1-10",
        },
        "description": "Fast, minimal evaluation",
        "category": "evaluation",
    },
    "domain_judge": {
        "type": "judge",
        "template": """As a {{domain}} expert, evaluate this response:

QUESTION: {{context}}
RESPONSE: {{response}}

{{domain_criteria}}

Rate from {{min_score}} to {{max_score}} based on {{domain}} standards.

{{domain_specific_guidance}}

Score: """,
        "variables": {
            "domain": "general",
            "domain_criteria": "Consider accuracy, completeness, and appropriateness for the domain.",
            "min_score": "1",
            "max_score": "10",
            "domain_specific_guidance": "Apply domain-specific quality standards.",
        },
        "description": "Domain-specific evaluation with expert perspective",
        "category": "evaluation",
    },
}

# Template categories for organization
TEMPLATE_CATEGORIES: Dict[str, str] = {
    "basic": "Fundamental templates for common use cases",
    "reasoning": "Templates that emphasize logical thinking and problem-solving",
    "professional": "Templates for professional and expert interactions",
    "learning": "Templates that incorporate learning patterns",
    "evaluation": "Templates for judging and scoring responses",
}

__all__ = ["UNIFIED_TEMPLATES", "TEMPLATE_CATEGORIES"]
