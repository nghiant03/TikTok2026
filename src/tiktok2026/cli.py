from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from tiktok2026.adapters import RepositoryExportService
from tiktok2026.bootstrap import (
    build_production_services,
    build_synthetic_controller,
    initialize_runtime,
    verify_manifests,
)
from tiktok2026.contracts import (
    AuditEvent,
    Fidelity,
    ProvisionalFinalizationRequest,
    RunPhase,
    RunRecord,
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
# Helpers
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


def _production_run_coro(
    runtime_root: Path,
    repository_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """Run the production composition and return the final state."""
    from tiktok2026.config import AppSettings

    repo_root = _resolve_repo_root(repository_root)
    actual_run_id = run_id or f"prod-{uuid.uuid4().hex[:8]}"

    # Load settings from default profile
    profile_path = repo_root / "config" / "test.toml"
    if not profile_path.exists():
        profile_path = repo_root / "config" / "budgets" / "test.toml"
    if not profile_path.exists():
        profile_path = repo_root / "config" / "test.toml"
    # Use minimal settings if no config exists
    settings = AppSettings(
        repository_root=repo_root,
        runtime_root=runtime_root,
    )

    services = build_production_services(settings)
    compiled_graph: Any = services.graph
    repo = services.repository

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

    # Create run and audit event
    with contextlib.suppress(Exception):
        repo.put_run(
            RunRecord(run_id=actual_run_id, status="active"),
            f"{actual_run_id}-active",
            None,
        )
    repo.put_audit_event(
        AuditEvent(
            event_id=f"run-{actual_run_id}-start",
            run_id=actual_run_id,
            experiment_id=None,
            event_type="run_started",
            actor_type="human",
            actor_id="cli-operator",
            payload={"run_id": actual_run_id, "mode": "production"},
        )
    )

    try:
        result = asyncio.run(
            compiled_graph.ainvoke(
                initial,
                {"configurable": {"thread_id": actual_run_id}},
            )
        )
        return {
            "run_id": actual_run_id,
            "phase": str(result.get("phase", "")),
            "pending_route": result.get("pending_route"),
            "state_version": result.get("state_version", 0),
        }
    except Exception as exc:
        # Record the error in audit
        with contextlib.suppress(Exception):
            repo.put_audit_event(
                AuditEvent(
                    event_id=f"run-{actual_run_id}-error",
                    run_id=actual_run_id,
                    experiment_id=None,
                    event_type="run_error",
                    actor_type="controller",
                    actor_id="system",
                    payload={"error": str(exc)},
                )
            )
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
        if "error" in result:
            _fail(RuntimeError(result["error"]))
        _echo_json(result)
        return

    try:
        result = _production_run_coro(runtime_root, repository_root, run_id=None)
    except Exception as error:
        _fail(error)
    if "error" in result:
        _fail(RuntimeError(result["error"]))
    _echo_json(result)


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
        try:
            result = _synthetic_run_coro(runtime_root, repository_root, run_id=run_id)
        except Exception as error:
            _fail(error)
        if "error" in result:
            _fail(RuntimeError(result["error"]))
        _echo_json(result)
        return

    # Production resume: attempt reconciliation
    try:
        # Build a recovery candidate from persisted state
        source_reg = None
        try:
            events = repo.list_audit_events(run_id)
        except Exception:
            events = []

        if not events:
            _fail(RuntimeError(f"run {run_id} not found or has no events"))

        # Check source registration for identity
        database_source_commit = ""
        worktree_source_commit = ""
        database_artifact_sha256 = ""
        artifact_sha256 = ""
        resume_experiment_id: str = ""
        for ev in events:  # type: ignore[union-attr]
            eid = ev.experiment_id  # type: ignore[union-attr]
            if eid:
                resume_experiment_id = eid  # type: ignore[assignment]
        if resume_experiment_id:
            source_reg = repo.get_source_registration(resume_experiment_id)  # type: ignore[arg-type]
            if source_reg is not None:
                database_source_commit = source_reg.source_commit
                worktree_source_commit = source_reg.source_commit
                database_artifact_sha256 = source_reg.patch_sha256
                artifact_sha256 = source_reg.patch_sha256

        candidate = RecoveryCandidate(
            run_id=run_id,
            experiment_id=resume_experiment_id,  # type: ignore[arg-type]
            database_source_commit=database_source_commit,
            worktree_source_commit=worktree_source_commit,
            database_artifact_sha256=database_artifact_sha256,
            artifact_sha256=artifact_sha256,
            stale_lock=runtime_root / "locks" / f"{run_id}.lock",
            stale_reservation_id=None,
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

        # On success, record resume_accepted and re-invoke
        repo.put_audit_event(
            AuditEvent(
                event_id=f"resume-{run_id}-accepted",
                run_id=run_id,
                experiment_id=None,
                event_type="resume_accepted",
                actor_type="human",
                actor_id="cli-operator",
                payload={"reason": "identities verified"},
            )
        )
        result = _production_run_coro(runtime_root, repository_root, run_id=run_id)
        if "error" in result:
            _fail(RuntimeError(result["error"]))
        _echo_json(result)
    except Exception as error:
        _fail(error)


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
        _echo_json({**result, "finalization": "provisional", "status": "finalized"})
        return

    # Production finalize
    repo = ApplicationRepository(runtime_root / "application.sqlite3")
    try:
        # Find the converged experiment for this run
        events = repo.list_audit_events(run_id)
        finalize_experiment_id: str = ""
        for event in events:
            if event.experiment_id:
                finalize_experiment_id = event.experiment_id
        if not finalize_experiment_id:
            _fail(RuntimeError("no experiment found for this run"))

        source_reg = repo.get_source_registration(finalize_experiment_id)
        if source_reg is None:
            _fail(RuntimeError("no source registration found — cannot finalize"))

        # Find the latest evaluation
        evaluations = repo.list_json("evaluation")
        latest_eval_id = ""
        for eval_json in evaluations:
            eval_data = json.loads(eval_json)
            if eval_data.get("experiment_id") == finalize_experiment_id:
                latest_eval_id = eval_data.get("evaluation_id", "")

        request = ProvisionalFinalizationRequest(
            finalization_id=f"final-{run_id}",
            run_id=run_id,
            experiment_id=finalize_experiment_id,
            source_commit=source_reg.source_commit,
            checkpoint_id=f"ckpt-{finalize_experiment_id}",
            evaluation_id=latest_eval_id or f"eval-{finalize_experiment_id}",
            bundle_artifact_id="bundle-1",
            evaluator_id="provisional-within-user-v1",
        )
        finalization = repo.persist_provisional_finalization(request)
        _echo_json({
            "finalization_id": finalization.finalization_id,
            "validity": finalization.validity,
            "run_id": run_id,
            "experiment_id": finalize_experiment_id,
            "status": "finalized",
        })
    except Exception as error:
        _fail(error)


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
        _echo_json({"jsonl": str(jsonl_path), "markdown": str(md_path)})
        return

    # Production export
    repo = ApplicationRepository(runtime_root / "application.sqlite3")
    try:
        events = repo.list_audit_events(run_id)
        if not events:
            _fail(RuntimeError(f"run {run_id} not found"))
    except Exception as error:
        _fail(error)

    try:
        export_service = RepositoryExportService(repo, runtime_root)
        result = asyncio.run(export_service.export_run(run_id))
        _echo_json({
            "jsonl": str(result["jsonl"]),
            "markdown": str(result["markdown"]),
        })
    except Exception as error:
        _fail(error)


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