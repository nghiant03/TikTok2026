import sqlite3
from pathlib import Path

import pytest

from tiktok2026.persistence.migrations import (
    MigrationChecksumError,
    MigrationRunner,
    application_migrations_path,
)


def test_changed_applied_migration_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "app.sqlite3"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    migration = migrations / "001_initial.sql"
    migration.write_text("CREATE TABLE sample (id TEXT PRIMARY KEY);", encoding="utf-8")
    MigrationRunner(database, migrations).apply()
    migration.write_text("CREATE TABLE changed (id TEXT PRIMARY KEY);", encoding="utf-8")
    with pytest.raises(MigrationChecksumError):
        MigrationRunner(database, migrations).apply()


def test_migrations_are_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "app.sqlite3"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_initial.sql").write_text(
        "CREATE TABLE sample (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    runner = MigrationRunner(database, migrations)
    runner.apply()
    runner.apply()


def test_failed_migration_rolls_back_all_schema_changes(tmp_path: Path) -> None:
    database = tmp_path / "app.sqlite3"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_broken.sql").write_text(
        "CREATE TABLE partial (id TEXT);\nCREATE TABLE partial (id TEXT);",
        encoding="utf-8",
    )

    with pytest.raises(sqlite3.OperationalError):
        MigrationRunner(database, migrations).apply()

    with sqlite3.connect(database) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='partial'"
        ).fetchone()
        applied = connection.execute("SELECT version FROM schema_migrations").fetchall()
    assert table is None
    assert applied == []


def test_criterion_history_migration_is_tracked_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "app.sqlite3"
    runner = MigrationRunner(database, application_migrations_path())
    runner.apply()
    runner.apply()

    with sqlite3.connect(database) as connection:
        applied = connection.execute(
            "SELECT version FROM schema_migrations WHERE version = 8"
        ).fetchall()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert applied == [(8,)]
    assert {
        "authority_validation_criterion_occurrences",
        "authority_validation_resolution_claims",
    } <= tables
