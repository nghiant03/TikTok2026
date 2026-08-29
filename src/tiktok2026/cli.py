from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from tiktok2026.bootstrap import initialize_runtime, verify_manifests
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.testing import run_synthetic_lifecycle

app = typer.Typer(no_args_is_help=True)


def _fail(error: Exception) -> NoReturn:
    typer.echo(str(error), err=True)
    raise typer.Exit(code=1)


@app.command("runtime-init")
def runtime_init(
    runtime_root: Annotated[Path, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    repository_root = repository_root or Path.cwd()
    try:
        services = initialize_runtime(repository_root, runtime_root)
    except Exception as error:
        _fail(error)
    typer.echo(str(services.paths.root))


@app.command("migrate")
def migrate(
    runtime_root: Annotated[Path, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    repository_root = repository_root or Path.cwd()
    try:
        services = initialize_runtime(repository_root, runtime_root)
        MigrationRunner(
            services.paths.application_db, repository_root / "migrations" / "application"
        ).apply()
        MigrationRunner(services.paths.graph_db, repository_root / "migrations" / "graph").apply()
    except Exception as error:
        _fail(error)
    typer.echo("migrations applied")


@app.command("verify-manifests")
def verify_manifest_command(
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    repository_root = repository_root or Path.cwd()
    try:
        manifest = verify_manifests(repository_root)
    except Exception as error:
        _fail(error)
    typer.echo(manifest.benchmark_id)


@app.command("synthetic-run")
def synthetic_run(
    iterations: Annotated[int, typer.Option(min=1)] = 2,
    runtime_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        result = asyncio.run(run_synthetic_lifecycle(iterations, runtime_root=runtime_root))
    except Exception as error:
        _fail(error)
    typer.echo(
        json.dumps(
            {
                "run_id": result.run_id,
                "experiment_ids": result.experiment_ids,
                "validity": result.finalization.validity,
                "jsonl": str(result.exports.jsonl),
                "markdown": str(result.exports.markdown),
            },
            indent=2,
        )
    )


@app.command("run")
def run() -> None:
    _fail(
        RuntimeError("production composition requires configured model, data, and Docker adapters")
    )


@app.command("resume")
def resume() -> None:
    _fail(RuntimeError("no recoverable production run was selected"))


@app.command("inspect")
def inspect_run(
    runtime_root: Annotated[Path, typer.Option()], run_id: Annotated[str, typer.Option()]
) -> None:
    repository = ApplicationRepository(runtime_root / "application.sqlite3")
    events = repository.list_audit_events(run_id)
    typer.echo(json.dumps([event.model_dump(mode="json") for event in events], default=str))


@app.command("finalize")
def finalize() -> None:
    _fail(RuntimeError("finalization requires an eligible converged production run"))


@app.command("export")
def export() -> None:
    _fail(RuntimeError("export requires a selected persisted run"))


@app.command("diagnostics")
def diagnostics(repository_root: Annotated[Path | None, typer.Option()] = None) -> None:
    repository_root = repository_root or Path.cwd()
    try:
        manifest = verify_manifests(repository_root)
    except Exception as error:
        _fail(error)
    typer.echo(
        json.dumps(
            {
                "protected_manifest": "verified",
                "evaluator_status": manifest.judging_evaluator_status,
                "live_checks": "not requested",
            }
        )
    )
