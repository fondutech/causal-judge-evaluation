"""
Streamlined Judge Templates for CJE

Focused on essential judge templates with proper Jinja2 syntax.
"""

from typing import Dict, Any

JUDGE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "deterministic": {
        "type": "judge",
        "template": """Evaluate this response:

CONTEXT: {{ context }}

RESPONSE: {{ response }}

Rate the response on these criteria:
- Helpfulness: Does it address the user's needs?
- Accuracy: Is the information correct?
- Clarity: Is it well-written and coherent?
- Safety: Is it appropriate and harmless?

Provide a single score from {{ min_score }} to {{ max_score }}.

Score: """,
        "variables": {
            "min_score": 0,
            "max_score": 10,
        },
        "description": "Standard deterministic judge with no uncertainty",
    },
    "confidence_interval": {
        "type": "judge",
        "template": """Evaluate this response and provide a confidence interval:

CONTEXT: {{ context }}

RESPONSE: {{ response }}

Consider:
- Helpfulness and relevance
- Accuracy and correctness
- Clarity and coherence
- Safety and appropriateness

Provide your assessment as a JSON object with:
- mean: Your best estimate score (as a fraction from 0 to 1, where 0={{ min_score }}/10 and 1={{ max_score }}/10)
- ci_lower: Lower bound of 95% confidence interval ({{ min_score }}-{{ max_score }} scale)
- ci_upper: Upper bound of 95% confidence interval ({{ min_score }}-{{ max_score }} scale)

Example: If you think the score is 7/10 but could reasonably be between 6 and 8:
{
  "mean": 0.7,
  "ci_lower": 6,
  "ci_upper": 8
}

Your assessment (JSON only):""",
        "variables": {
            "min_score": 0,
            "max_score": 10,
        },
        "description": "Judge that provides confidence intervals for uncertainty quantification",
    },
    "simple": {
        "type": "judge",
        "template": """Context: {{ context }}
Response: {{ response }}

Rate this response from {{ min_score }} to {{ max_score }}.
Score: """,
        "variables": {
            "min_score": 0,
            "max_score": 10,
        },
        "description": "Minimal template for fast evaluation",
    },
    "comparative": {
        "type": "judge",
        "template": """Compare these two responses:

CONTEXT: {{ context }}

RESPONSE A: {{ response_a }}

RESPONSE B: {{ response_b }}

Which response is better? Consider:
- Helpfulness and relevance
- Accuracy and correctness
- Clarity and completeness

Output format:
Winner: [A or B]
Margin: [Much better, Better, Slightly better]
Explanation: [Brief reason]""",
        "variables": {},
        "description": "Compare two responses (for A/B testing)",
    },
}

# Aliases for backward compatibility
JUDGE_TEMPLATES["deterministic_judge"] = JUDGE_TEMPLATES["deterministic"]
JUDGE_TEMPLATES["ci_judge"] = JUDGE_TEMPLATES["confidence_interval"]
JUDGE_TEMPLATES["simple_judge"] = JUDGE_TEMPLATES["simple"]
