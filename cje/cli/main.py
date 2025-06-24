"""CJE command-line interface main entry point."""

import importlib.metadata as im
import typer
from .decode_and_log import app as log_app
from .calibrate import app as calib_app
from .estimate import app as est_app
from .run import run as run_cmd
from .judge import app as judge_app
from .backfill_logp import app as backfill_app
from .validate import app as validate_app

app = typer.Typer(help="CJE command-line interface")
app.add_typer(log_app, name="log")
app.add_typer(calib_app, name="calibrate")
app.add_typer(est_app, name="estimate")
app.command("run")(run_cmd)
app.add_typer(judge_app, name="judge")
app.add_typer(backfill_app, name="backfill")
app.add_typer(validate_app, name="validate")


@app.command()
def version() -> None:
    """Print version."""
    print(im.version("cje"))


if __name__ == "__main__":
    app()
