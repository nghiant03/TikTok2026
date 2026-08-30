from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from dotenv import load_dotenv
from typer import Typer

from tiktok2026.bootstrap import (
    ProductionOperations,
    build_production_operations,
)
from tiktok2026.contracts import OperationResult
from tiktok2026.logging import configure_logging

load_dotenv()

app = Typer(no_args_is_help=True)


def _fail(error: Exception) -> NoReturn:
    typer.echo(str(error), err=True)
    raise typer.Exit(code=1)


def _operations(
    runtime_root: Path,
    repository_root: Path | None,
    profile_path: Path | None = None,
    operator_config: Path | None = None,
) -> ProductionOperations:
    configure_logging(runtime_root)
    return build_production_operations(
        repository_root or Path.cwd(), runtime_root, profile_path, operator_config
    )


def _render(result: OperationResult) -> None:
    typer.echo(json.dumps(result.model_dump(mode="json"), default=str, indent=2))


@app.command("runtime-init")
def runtime_init(
    runtime_root: Annotated[Path, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        result = _operations(runtime_root, repository_root).runtime_init()
    except Exception as error:
        _fail(error)
    typer.echo(str(result.values["root"]))


@app.command("migrate")
def migrate(
    runtime_root: Annotated[Path, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        _operations(runtime_root, repository_root).migrate()
    except Exception as error:
        _fail(error)
    typer.echo("migrations applied")


@app.command("verify-manifests")
def verify_manifest_command(
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        result = build_production_operations(
            repository_root or Path.cwd(), repository_root or Path.cwd()
        ).verify_manifests()
    except Exception as error:
        _fail(error)
    typer.echo(str(result.values["benchmark_id"]))


@app.command("synthetic-run")
def synthetic_run(
    iterations: Annotated[int, typer.Option(min=1)] = 2,
    runtime_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    root = runtime_root or Path.cwd().parent / f"{Path.cwd().name}.runtime"
    try:
        result = _operations(root, Path.cwd()).synthetic_run(iterations)
    except Exception as error:
        _fail(error)
    _render(result)


@app.command("calibrate-baseline")
def calibrate_baseline_command(
    runtime_root: Annotated[Path, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    profile_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        result = _operations(runtime_root, repository_root, profile_path).calibrate_baseline()
    except Exception as error:
        _fail(error)
    _render(result)


@app.command("run")
def run_command(
    runtime_root: Annotated[Path, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    profile_path: Annotated[Path | None, typer.Option()] = None,
    operator_config: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[bool, typer.Option("--synthetic")] = False,
) -> None:
    try:
        result = _operations(
            runtime_root, repository_root, profile_path, operator_config
        ).run(synthetic=synthetic)
    except Exception as error:
        _fail(error)
    _render(result)


@app.command("resume")
def resume_command(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    profile_path: Annotated[Path | None, typer.Option()] = None,
    operator_config: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[bool, typer.Option("--synthetic")] = False,
) -> None:
    try:
        result = _operations(
            runtime_root, repository_root, profile_path, operator_config
        ).resume(run_id, synthetic=synthetic)
    except Exception as error:
        _fail(error)
    _render(result)


@app.command("inspect")
def inspect_run(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
) -> None:
    try:
        result = _operations(runtime_root, None).inspect(run_id)
    except Exception as error:
        _fail(error)
    _render(result)


@app.command("finalize")
def finalize_command(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[bool, typer.Option("--synthetic")] = False,
) -> None:
    del synthetic
    try:
        result = _operations(runtime_root, repository_root).finalize(run_id)
    except Exception as error:
        _fail(error)
    _render(result)


@app.command("export")
def export_command(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[bool, typer.Option("--synthetic")] = False,
) -> None:
    del synthetic
    try:
        result = _operations(runtime_root, repository_root).export(run_id)
    except Exception as error:
        _fail(error)
    _render(result)


@app.command("diagnostics")
def diagnostics(
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
    root = repository_root or Path.cwd()
    try:
        result = build_production_operations(root, root).diagnostics()
    except Exception as error:
        _fail(error)
    _render(result)
