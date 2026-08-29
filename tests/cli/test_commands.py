import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from tiktok2026.cli import app

ROOT = Path(__file__).parents[2]


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
    assert versions == [(1,), (2,), (3,)]


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
