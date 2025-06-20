"""CLI for judge functionality."""

import json, pathlib, typer
from enum import Enum
from rich import print
from typing import Dict, Any
from ..judge import JudgeFactory
from jinja2 import Template
from ..prompts import UNIFIED_TEMPLATES


class JudgeType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    PROMETHEUS = "prometheus"
    HUGGINGFACE = "huggingface"


app = typer.Typer(help="Judge functionality")

# Get available judge templates
JUDGE_TEMPLATES = {
    name: template["template"]
    for name, template in UNIFIED_TEMPLATES.items()
    if template.get("type") == "judge"
}


@app.command()
def run(
    log_file: pathlib.Path = typer.Option(...),
    judge: JudgeType = typer.Option(
        ..., help="openai | anthropic | prometheus | huggingface"
    ),
    template: str = typer.Option("quick_judge", help="Prompt template id"),
    out_jsonl: pathlib.Path = typer.Option(..., help="Write JSONL with `score_raw`"),
    model_name: str = typer.Option(None, help="Override default model name"),
) -> None:
    """Attach baseline judge scores to a JSONL log."""
    rows = [json.loads(l) for l in log_file.read_text().splitlines()]

    # Create judge using modern factory system
    judge_kwargs: Dict[str, Any] = {"template": template}

    # Handle both string and enum inputs for judge
    # When called programmatically (not via CLI), judge might be a string
    if isinstance(judge, str):
        judge_name = judge
    else:
        judge_name = judge.value

    # Map judge types to providers and default models
    judge_configs = {
        "openai": {"provider": "openai", "model": model_name or "gpt-4o-mini"},
        "anthropic": {
            "provider": "anthropic",
            "model": model_name or "claude-3-haiku-20240307",
        },
        "prometheus": {
            "provider": "prometheus",
            "model": model_name or "prometheus-eval/prometheus-13b-v1.0",
        },
        "huggingface": {
            "provider": "huggingface",
            "model": model_name or "microsoft/DialoGPT-medium",
        },
    }

    if judge_name not in judge_configs:
        raise ValueError(f"Unknown judge type: {judge_name}")

    config = judge_configs[judge_name]

    # For API judges (openai, anthropic), use the explicit create method
    if judge_name in ["openai", "anthropic"]:
        judge_instance = JudgeFactory.create(
            provider=config["provider"],
            model=config["model"],
            template=template,  # Pass template directly, not in kwargs
        )
    else:
        # For local judges (prometheus, huggingface), we need to handle differently
        # Since the factory only supports API judges now
        raise NotImplementedError(
            f"Local judge '{judge_name}' is not supported in the new API"
        )

    # Prepare samples for batch scoring
    samples = [{"context": row["context"], "response": row["response"]} for row in rows]

    # Score all samples
    scores = judge_instance.score_batch(samples)

    # Add scores to rows using unified storage
    from ..utils.score_storage import update_row_with_score

    updated_rows = []
    for row, score in zip(rows, scores):
        updated_row = update_row_with_score(row, score, "score_raw")
        updated_rows.append(updated_row)
    rows = updated_rows

    out_jsonl.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"Wrote {out_jsonl}")
