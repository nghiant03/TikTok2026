from pathlib import Path

import pytest

from tiktok2026.persistence.migrations import MigrationChecksumError, MigrationRunner


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

    with pytest.raises(Exception):
        MigrationRunner(database, migrations).apply()

    import sqlite3

    with sqlite3.connect(database) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='partial'"
        ).fetchone()
        applied = connection.execute("SELECT version FROM schema_migrations").fetchall()
    assert table is None
    assert applied == []
