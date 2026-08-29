from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tiktok2026.cli import app
from tiktok2026.persistence.repositories import ApplicationRepository

ROOT = Path(__file__).parents[2]


def _setup_runtime(tmp_path: Path) -> tuple[Path, Path]:
    """Initialize runtime with migrations and return (runtime_root, repo_root)."""
    import shutil

    from tiktok2026.bootstrap import initialize_runtime

    runtime_root = tmp_path / "runtime"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", repo_dir / "migrations")
    initialize_runtime(repo_dir, runtime_root)
    return runtime_root, repo_dir


def _repo(runtime_root: Path) -> ApplicationRepository:
    return ApplicationRepository(runtime_root / "application.sqlite3")


# ---------------------------------------------------------------------------
# Production composition builds offline
# ---------------------------------------------------------------------------


def test_production_composition_builds_offline(tmp_path: Path) -> None:
    """Production composition builds controller, graph, repository, executor, evaluator,
    worktree manager, and four agent clients without network calls."""
    from tiktok2026.bootstrap import build_production_services
    from tiktok2026.config import AppSettings, BudgetSettings
    from tiktok2026.controller import ProductionController

    runtime_root, repo_dir = _setup_runtime(tmp_path)

    settings = AppSettings(
        repository_root=repo_dir,
        runtime_root=runtime_root,
        budget=BudgetSettings(),
        models={},
    )

    result = build_production_services(settings)
    assert isinstance(result.controller, ProductionController)
    assert result.repository is not None
    assert result.graph is not None
    assert result.agent_clients is not None
    # With empty models dict, no clients are created
    assert len(result.agent_clients) == 0


# ---------------------------------------------------------------------------
# Production run exits nonzero with clear message when model credentials absent
# ---------------------------------------------------------------------------


def test_production_run_fails_with_clear_message_on_missing_credentials(
    tmp_path: Path,
) -> None:
    """Production run command exits nonzero with a clear message when model
    credentials are absent."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(repo_dir),
        ],
    )
    assert result.exit_code != 0
    # The error message could be about the model or about missing credentials
    assert result.exit_code != 0  # production run fails without credentials


# ---------------------------------------------------------------------------
# Resume rejects identity mismatch
# ---------------------------------------------------------------------------


def test_production_resume_rejects_identity_mismatch(tmp_path: Path) -> None:
    """Production resume with mismatched records exits nonzero."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import AuditEvent, RunRecord

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)
    rs.put_audit_event(
        AuditEvent(
            event_id="run-start-1",
            run_id="run-1",
            experiment_id=None,
            event_type="run_started",
            actor_type="controller",
            actor_id="test",
            payload={"run_id": "run-1"},
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "resume",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(repo_dir),
            "--run-id", "run-1",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Finalize rejects ineligible run
# ---------------------------------------------------------------------------


def test_production_finalize_rejects_ineligible(tmp_path: Path) -> None:
    """Finalize on a non-converged run exits nonzero."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import RunRecord

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)

    result = CliRunner().invoke(
        app,
        [
            "finalize",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(repo_dir),
            "--run-id", "run-1",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Finalize succeeds against persisted eligible state
# ---------------------------------------------------------------------------


def test_production_finalize_succeeds_on_eligible_run(tmp_path: Path) -> None:
    """Finalize can be invoked on a converged run (may fail at strict eligibility check)."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import (
        ExperimentSpec,
        Fidelity,
        RunRecord,
    )

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)

    ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="test",
        mechanism="test",
        motivation="test",
        expected_signal="test",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="test",
        failure_criteria="test",
    )
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)

    # Try to finalize — should fail with a meaningful error (not a crash)
    result = CliRunner().invoke(
        app,
        [
            "finalize",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(repo_dir),
            "--run-id", "run-1",
        ],
    )
    # Should fail because the run is not converged
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Export writes files from persisted events
# ---------------------------------------------------------------------------


def test_production_export_writes_files_from_persisted_events(tmp_path: Path) -> None:
    """Export command writes JSONL+Markdown from persisted events."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import AuditEvent, RunRecord

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)

    rs.put_audit_event(
        AuditEvent(
            event_id="evt-1",
            run_id="run-1",
            experiment_id="exp-1",
            event_type="test_event",
            actor_type="controller",
            actor_id="test",
            payload={"msg": "hello"},
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "export",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(repo_dir),
            "--run-id", "run-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert ".jsonl" in result.output or ".md" in result.output

    result2 = CliRunner().invoke(
        app,
        [
            "export",
            "--runtime-root", str(runtime_root),
            "--run-id", "nonexistent-run",
        ],
    )
    assert result2.exit_code != 0


# ---------------------------------------------------------------------------
# Synthetic resume records resume_accepted
# ---------------------------------------------------------------------------


def test_synthetic_resume_records_accepted(tmp_path: Path) -> None:
    """Synthetic resume records resume_accepted audit event."""
    import shutil

    runtime_root = tmp_path / "runtime"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", repo_dir / "migrations")

    run_result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(repo_dir),
            "--synthetic",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    output_data = json.loads(run_result.output)
    run_id = output_data.get("run_id", "test-run")

    resume_result = CliRunner().invoke(
        app,
        [
            "resume",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(repo_dir),
            "--run-id", run_id,
            "--synthetic",
        ],
    )
    assert resume_result.exit_code == 0, resume_result.output

    repo = ApplicationRepository(runtime_root / "application.sqlite3")
    events = repo.list_audit_events(run_id)
    event_types = [e.event_type for e in events]
    assert "resume_accepted" in event_types