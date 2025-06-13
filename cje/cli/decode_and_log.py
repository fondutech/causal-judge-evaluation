import json, typer, pathlib
from rich.progress import track
from ..data import load_dataset, DatasetName
from ..loggers.policy import PolicyRunner

app = typer.Typer(help="Generate responses under π₀ and write log JSONL")


@app.command()
def run(
    model: str = typer.Option(..., help="HF checkpoint for logging policy π₀"),
    dataset: DatasetName = typer.Option(..., help="Dataset name or path to data file"),
    split: str = typer.Option("train"),
    out: pathlib.Path = typer.Option(..., help="Output JSONL"),
    max_new_tokens: int = typer.Option(128, help="Maximum new tokens to generate"),
    temperature: float = typer.Option(0.0, help="Sampling temperature"),
    top_p: float = typer.Option(1.0, help="Nucleus sampling probability"),
) -> None:
    ds = load_dataset(dataset, split=split)
    pr = PolicyRunner(
        model, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )
    with out.open("w") as fh:
        for samp in track(ds.itersamples(), description="Decoding"):
            ctx = samp.context
            result = pr.generate_with_logp([ctx], return_token_logprobs=True)[0]
            if len(result) == 3:
                text, logp_tok, token_logps = result
            else:
                text, logp_tok = result
                token_logps = []
            row = {
                "uid": samp.uid,
                "context": samp.context,
                "response": text,
                "logp": float(logp_tok),
                "token_logp": token_logps,
                "y_true": samp.y_true,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            fh.write(json.dumps(row) + "\n")
