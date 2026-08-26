import asyncio
import json

import typer

from tiktok2026.testing import run_synthetic_lifecycle

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command("synthetic-run")
def synthetic_run(iterations: int = typer.Option(2, min=1)) -> None:
    result = asyncio.run(run_synthetic_lifecycle(iterations))
    typer.echo(json.dumps(result, default=str, indent=2))
