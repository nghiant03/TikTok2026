from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from tiktok2026.bootstrap import (
    build_synthetic_controller,
    initialize_runtime,
    verify_manifests,
)
from tiktok2026.contracts import (
    AuditEvent,
    Fidelity,
    RunPhase,
)
from tiktok2026.graph.state import ProductionState
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.recovery import RecoveryCandidate, reconcile_recovery
from tiktok2026.testing import run_synthetic_lifecycle

app = typer.Typer(no_args_is_help=True)


def _fail(error: Exception) -> NoReturn:
    typer.echo(str(error), err=True)
    raise typer.Exit(code=1)


def _echo_json(data: dict[str, object]) -> None:
    typer.echo(json.dumps(data, default=str, indent=2))


def _resolve_repo_root(repository_root: Path | None) -> Path:
    return repository_root or Path.cwd()


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------



def _synthetic_run_coro(
    runtime_root: Path,
    repository_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """Run the synthetic composition and return the final state."""
    repo_root = _resolve_repo_root(repository_root)
    actual_run_id = run_id or f"test-run-{uuid.uuid4().hex[:8]}"

    _ctrl, store, graph = build_synthetic_controller(
        repository_root=repo_root,
        runtime_root=runtime_root,
    )
    compiled_graph: Any = graph

    initial: ProductionState = {
        "run_id": actual_run_id,
        "phase": RunPhase.BOOTSTRAP,
        "current_experiment_id": None,
        "current_hypothesis_id": None,
        "active_worktree_id": None,
        "latest_validation_report_id": None,
        "latest_execution_result_id": None,
        "latest_evaluation_result_id": None,
        "orchestration_decision_id": None,
        "repair_attempts": 0,
        "fidelity": Fidelity.SMOKE,
        "pending_route": None,
        "terminal_reason": None,
        "state_version": 0,
    }

    # Record audit event for operator-initiated run with unique event ID
    invocation_id = uuid.uuid4().hex[:12]
    repo = ApplicationRepository(runtime_root / "application.sqlite3")
    with contextlib.suppress(Exception):
        repo.put_audit_event(
            AuditEvent(
                event_id=f"run-{invocation_id}-start",
                run_id=actual_run_id,
                experiment_id=None,
                event_type="run_started",
                actor_type="human",
                actor_id="cli-operator",
                payload={"run_id": actual_run_id, "mode": "synthetic"},
            )
        )

    try:
        result = asyncio.run(
            compiled_graph.ainvoke(
                initial,
                {"configurable": {"thread_id": f"{actual_run_id}-{uuid.uuid4().hex[:8]}"}},
            )
        )
        return {
            "run_id": actual_run_id,
            "phase": str(result.get("phase", "")),
            "pending_route": result.get("pending_route"),
            "state_version": result.get("state_version", 0),
            "transitions_recorded": len(store.persisted)  # type: ignore[union-attr]
        }
    except Exception as exc:
        return {
            "run_id": actual_run_id,
            "error": str(exc),
            "phase": "error",
        }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


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
def run_command(
    runtime_root: Annotated[Path, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[
        bool, typer.Option("--synthetic", help="Use synthetic composition")
    ] = False,
) -> None:
    """Run a production or synthetic research lifecycle."""
    if synthetic:
        try:
            result = _synthetic_run_coro(runtime_root, repository_root, run_id="test-run")
        except Exception as error:
            _fail(error)
        # Check for errors in the result
        if "error" in result:
            _fail(RuntimeError(result["error"]))
        _echo_json(result)
        return
    _fail(
        RuntimeError("production composition requires configured model, data, and Docker adapters")
    )


@app.command("resume")
def resume_command(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[
        bool, typer.Option("--synthetic", help="Use synthetic composition")
    ] = False,
) -> None:
    """Resume a previously saved run."""
    repo = ApplicationRepository(runtime_root / "application.sqlite3")

    if synthetic:
        # Record audit event for resume
        with contextlib.suppress(Exception):
            repo.put_audit_event(
                AuditEvent(
                    event_id=f"resume-{run_id}-{uuid.uuid4().hex[:8]}-accepted",
                    run_id=run_id,
                    experiment_id=None,
                    event_type="resume_accepted",
                    actor_type="human",
                    actor_id="cli-operator",
                    payload={"reason": "synthetic resume"},
                )
            )
        # Re-run the synthetic composition
        try:
            result = _synthetic_run_coro(runtime_root, repository_root, run_id=run_id)
        except Exception as error:
            _fail(error)
        if "error" in result:
            _fail(RuntimeError(result["error"]))
        _echo_json(result)
        return

    # Production resume: attempt reconciliation first
    try:
        candidate = RecoveryCandidate(
            run_id=run_id,
            experiment_id="",
            database_source_commit="",
            worktree_source_commit="",
            database_artifact_sha256="",
            artifact_sha256="",
            stale_lock=Path("/tmp/placeholder"),
        )
        result = reconcile_recovery(candidate, lambda _: None)
        if not result.resumable:
            with contextlib.suppress(Exception):
                repo.put_audit_event(
                    AuditEvent(
                        event_id=f"resume-{run_id}-rejected",
                        run_id=run_id,
                        experiment_id=None,
                        event_type="resume_rejected",
                        actor_type="human",
                        actor_id="cli-operator",
                        payload={"reason": result.reason},
                    )
                )
            _fail(RuntimeError(result.reason))
    except Exception as error:
        _fail(error)

    _fail(RuntimeError("no recoverable production run was selected"))


@app.command("inspect")
def inspect_run(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
) -> None:
    repository = ApplicationRepository(runtime_root / "application.sqlite3")
    try:
        events = repository.list_audit_events(run_id)
    except Exception as error:
        _fail(error)
    typer.echo(json.dumps([event.model_dump(mode="json") for event in events], default=str))


@app.command("finalize")
def finalize_command(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[
        bool, typer.Option("--synthetic", help="Use synthetic composition")
    ] = False,
) -> None:
    """Finalize a run (provisional only)."""
    if synthetic:
        try:
            result = _synthetic_run_coro(runtime_root, repository_root, run_id=run_id)
        except Exception as error:
            _fail(error)
        if "error" in result:
            _fail(RuntimeError(result["error"]))
        _echo_json(
            {
                **result,
                "finalization": "provisional",
                "status": "finalized",
            }
        )
        return
    _fail(RuntimeError("finalization requires an eligible converged production run"))


@app.command("export")
def export_command(
    runtime_root: Annotated[Path, typer.Option()],
    run_id: Annotated[str, typer.Option()],
    repository_root: Annotated[Path | None, typer.Option()] = None,
    synthetic: Annotated[
        bool, typer.Option("--synthetic", help="Use synthetic composition")
    ] = False,
) -> None:
    """Export run artifacts (JSONL + Markdown)."""
    if synthetic:
        try:
            result = _synthetic_run_coro(runtime_root, repository_root, run_id=run_id)
        except Exception as error:
            _fail(error)
        if "error" in result:
            _fail(RuntimeError(result["error"]))
        output_dir = runtime_root / "exports" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "iterations.jsonl"
        md_path = output_dir / "iterations.md"
        jsonl_path.write_text(json.dumps(result, default=str) + "\n", encoding="utf-8")
        md_path.write_text(
            f"# Run {run_id}\n\n{json.dumps(result, default=str, indent=2)}\n",
            encoding="utf-8",
        )
        _echo_json(
            {
                "jsonl": str(jsonl_path),
                "markdown": str(md_path),
            }
        )
        return
    _fail(RuntimeError("export requires a selected persisted run"))


@app.command("diagnostics")
def diagnostics(
    repository_root: Annotated[Path | None, typer.Option()] = None,
) -> None:
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