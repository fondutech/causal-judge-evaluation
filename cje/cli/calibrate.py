import json
import pathlib
import typer
import numpy as np
from rich.progress import track
from ..calibration.isotonic import fit_isotonic, plot_reliability
from ..calibration.cross_fit import cross_fit_calibration

app = typer.Typer(help="Fit isotonic calibration on a JSONL log file")


@app.command()
def run(
    log_file: pathlib.Path = typer.Option(
        ..., help="Input JSONL with raw judge scores"
    ),
    score_key: str = typer.Option("score_raw", help="JSON field with raw score"),
    label_key: str = typer.Option(
        "y_true", help="JSON field with ground-truth utility"
    ),
    out_file: pathlib.Path = typer.Option(..., help="Write calibrated JSONL"),
    curve_png: pathlib.Path = typer.Option(..., help="Reliability curve PNG"),
) -> None:
    """Reads log_file, fits monotone gφ, writes new file with `score_cal`."""
    rows = [json.loads(l) for l in log_file.read_text().splitlines()]
    scores = np.array([r[score_key] for r in rows], dtype=float)
    y_true = np.array([r[label_key] for r in rows], dtype=float)
    iso = fit_isotonic(scores, y_true)
    for r, s in zip(rows, scores):
        r["score_cal"] = float(iso.predict([s])[0])
    out_file.write_text("\n".join(json.dumps(r) for r in rows))
    plot_reliability(scores, y_true, iso, curve_png)
    typer.echo(f"Calibrated file → {out_file}\nCurve → {curve_png}")


@app.command()
def cross_fit(
    log_file: pathlib.Path = typer.Option(
        ..., help="Input JSONL with raw judge scores"
    ),
    out_file: pathlib.Path = typer.Option(..., help="Write calibrated JSONL"),
    k_folds: int = typer.Option(5, help="Number of cross-validation folds"),
    seed: int = typer.Option(42, help="Random seed for fold assignment"),
    score_key: str = typer.Option("score_raw", help="JSON field with raw score"),
    label_key: str = typer.Option(
        "y_true", help="JSON field with ground-truth utility"
    ),
    curve_png: pathlib.Path = typer.Option(None, help="Optional reliability curve PNG"),
) -> None:
    """Cross-fitted calibration to avoid data leakage."""
    from rich.console import Console

    console = Console()

    # Load data
    with console.status("[bold blue]Loading data..."):
        rows = [json.loads(l) for l in log_file.read_text().splitlines()]
        console.print(f"[green]✓[/green] Loaded {len(rows)} rows")

    # Run cross-fit calibration
    with console.status("[bold blue]Running cross-fit calibration..."):
        calibrated_rows, diagnostics = cross_fit_calibration(
            rows,
            k_folds=k_folds,
            seed=seed,
            score_key=score_key,
            label_key=label_key,
            plot_path=curve_png,
        )

    # Save results
    out_file.write_text("\n".join(json.dumps(r) for r in calibrated_rows))

    # Print diagnostics
    console.print("\n[bold green]Calibration Complete![/bold green]")
    console.print(f"Oracle rows: {diagnostics['n_oracle']}")
    console.print(f"RMSE reduction: {diagnostics['rmse_reduction']:.1f}%")
    console.print(f"Coverage at ±0.1: {diagnostics['coverage_at_0.1']:.1%}")

    if "fold_stats" in diagnostics:
        console.print("\n[bold]Per-fold statistics:[/bold]")
        for fold_name, stats in diagnostics["fold_stats"].items():
            console.print(
                f"  {fold_name}: n={stats['n']}, "
                f"RMSE={stats['rmse']:.3f}, "
                f"coverage={stats['coverage']:.1%}"
            )

    console.print(f"\n[green]✓[/green] Calibrated file → {out_file}")
    if curve_png:
        console.print(f"[green]✓[/green] Reliability curve → {curve_png}")
