from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from tiktok2026.cli import app

ROOT = Path(__file__).parents[2]


def test_operator_commands_are_registered() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "runtime-init",
        "migrate",
        "verify-manifests",
        "synthetic-run",
        "run",
        "resume",
        "inspect",
        "finalize",
        "export",
        "diagnostics",
    ):
        assert command in result.output


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


def test_run_synthetic_exits_zero(tmp_path: Path) -> None:
    """run command with synthetic runtime completes successfully."""
    import shutil

    runtime_root = tmp_path / "runtime"
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", test_repo_root / "migrations")

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(test_repo_root),
            "--synthetic",
        ],
    )
    assert result.exit_code == 0, result.output
    output_data = json.loads(result.output)
    assert output_data.get("phase") == "complete"
    # Verify graph DB was created
    assert (runtime_root / "graph.sqlite3").exists()


def test_run_synthetic_without_required_option_fails() -> None:
    """run command without --runtime-root exits nonzero."""
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# resume command
# ---------------------------------------------------------------------------


def test_resume_on_clean_synthetic_run_succeeds(tmp_path: Path) -> None:
    """resume after a completed synthetic run reaches export."""
    import shutil

    runtime_root = tmp_path / "runtime"
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", test_repo_root / "migrations")

    # First run the synthetic run
    run_result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(test_repo_root),
            "--synthetic",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    # Parse the run_id from the output
    output_data = json.loads(run_result.output)
    run_id = output_data.get("run_id", "test-run")

    # Now resume
    resume_result = CliRunner().invoke(
        app,
        [
            "resume",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(test_repo_root),
            "--run-id", run_id,
            "--synthetic",
        ],
    )
    assert resume_result.exit_code == 0, resume_result.output


def test_resume_rejects_nonresumable_state(tmp_path: Path) -> None:
    """resume with a non-existent run-id exits nonzero."""
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    result = CliRunner().invoke(
        app,
        [
            "resume",
            "--runtime-root", str(runtime_root),
            "--run-id", "nonexistent-run",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# finalize command
# ---------------------------------------------------------------------------


def test_finalize_on_synthetic_run_succeeds(tmp_path: Path) -> None:
    """finalize after a completed synthetic run succeeds."""
    import shutil

    runtime_root = tmp_path / "runtime"
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", test_repo_root / "migrations")

    # Run first
    run_result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(test_repo_root),
            "--synthetic",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    output_data = json.loads(run_result.output)
    run_id = output_data.get("run_id", "test-run")

    # Finalize
    result = CliRunner().invoke(
        app,
        [
            "finalize",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(test_repo_root),
            "--run-id", run_id,
            "--synthetic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "provisional" in result.output.lower()


def test_finalize_rejects_nonexistent_run(tmp_path: Path) -> None:
    """finalize on a nonexistent run exits nonzero."""
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    result = CliRunner().invoke(
        app,
        [
            "finalize",
            "--runtime-root", str(runtime_root),
            "--run-id", "nonexistent",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# export command
# ---------------------------------------------------------------------------


def test_export_on_synthetic_run_succeeds(tmp_path: Path) -> None:
    """export after a completed synthetic run produces files."""
    import shutil

    runtime_root = tmp_path / "runtime"
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", test_repo_root / "migrations")

    # Run first
    run_result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(test_repo_root),
            "--synthetic",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    output_data = json.loads(run_result.output)
    run_id = output_data.get("run_id", "test-run")

    # Export
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--runtime-root", str(runtime_root),
            "--repository-root", str(test_repo_root),
            "--run-id", run_id,
            "--synthetic",
        ],
    )
    assert result.exit_code == 0, result.output
    # Should print paths to exported files
    assert ".jsonl" in result.output or ".md" in result.output


def test_export_rejects_nonexistent_run(tmp_path: Path) -> None:
    """export on a nonexistent run exits nonzero."""
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    result = CliRunner().invoke(
        app,
        [
            "export",
            "--runtime-root", str(runtime_root),
            "--run-id", "nonexistent",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Recovery tests
# ---------------------------------------------------------------------------


def test_recovery_releases_stale_state_when_all_identities_agree(tmp_path: Path) -> None:
    from tiktok2026.recovery import RecoveryCandidate, reconcile_recovery

    lock = tmp_path / "run.lock"
    lock.write_text("run-1", encoding="utf-8")
    candidate = RecoveryCandidate(
        run_id="run-1",
        experiment_id="experiment-1",
        database_source_commit="a" * 40,
        worktree_source_commit="a" * 40,
        database_artifact_sha256="b" * 64,
        artifact_sha256="b" * 64,
        stale_lock=lock,
        stale_reservation_id="reservation-1",
    )
    released: list[str] = []

    result = reconcile_recovery(candidate, released.append)

    assert result.resumable
    assert result.released_reservation_id == "reservation-1"
    assert not lock.exists()
    assert released == ["reservation-1"]


def test_recovery_preserves_state_and_blocks_on_identity_mismatch(tmp_path: Path) -> None:
    from tiktok2026.recovery import RecoveryCandidate, reconcile_recovery

    lock = tmp_path / "run.lock"
    lock.write_text("run-1", encoding="utf-8")
    candidate = RecoveryCandidate(
        run_id="run-1",
        experiment_id="experiment-1",
        database_source_commit="a" * 40,
        worktree_source_commit="c" * 40,
        database_artifact_sha256="b" * 64,
        artifact_sha256="b" * 64,
        stale_lock=lock,
        stale_reservation_id="reservation-1",
    )
    released: list[str] = []

    result = reconcile_recovery(candidate, released.append)

    assert not result.resumable
    assert result.reason == "source identity mismatch"
    assert lock.exists()
    assert released == []


def test_runtime_init_creates_external_layout_and_applies_actual_migrations(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    result = CliRunner().invoke(
        app,
        [
            "runtime-init",
            "--repository-root",
            str(ROOT),
            "--runtime-root",
            str(runtime_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (runtime_root / "application.sqlite3").exists()
    assert (runtime_root / "graph.sqlite3").exists()
    with sqlite3.connect(runtime_root / "application.sqlite3") as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert {"experiments", "audit_events", "evaluations", "finalizations"} <= tables
    assert versions == [(1,), (2,), (3,), (4,), (5,)]


def test_runtime_init_rejects_runtime_root_inside_repository() -> None:
    result = CliRunner().invoke(
        app,
        [
            "runtime-init",
            "--repository-root",
            str(ROOT),
            "--runtime-root",
            str(ROOT / "runtime"),
        ],
    )

    assert result.exit_code != 0
    assert "outside the repository" in result.output