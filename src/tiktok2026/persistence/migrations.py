from __future__ import annotations

import fcntl
import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class MigrationChecksumError(RuntimeError):
    pass


def application_migrations_path() -> Path:
    return Path(__file__).resolve().parents[3] / "migrations" / "application"


class MigrationRunner:
    def __init__(self, database: Path, migrations: Path) -> None:
        self.database = database
        self.migrations = migrations

    def apply(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        lock = self.database.with_name(f"{self.database.name}.migration.lock")
        with lock.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                self._apply_locked()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _apply_locked(self) -> None:
        paths = sorted(self.migrations.glob("[0-9][0-9][0-9]_*.sql"))
        first_sql = paths[0].read_text(encoding="utf-8") if paths else ""
        with sqlite3.connect(self.database) as connection:
            tracking_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            if not tracking_exists and "CREATE TABLE schema_migrations" not in first_sql:
                connection.execute(
                    "CREATE TABLE schema_migrations ("
                    "version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, "
                    "applied_at TEXT NOT NULL)"
                )
                connection.commit()
            applied = (
                dict(
                    connection.execute("SELECT version, checksum FROM schema_migrations").fetchall()
                )
                if tracking_exists or "CREATE TABLE schema_migrations" not in first_sql
                else {}
            )
            for path in paths:
                version = int(path.name.split("_", 1)[0])
                sql = path.read_text(encoding="utf-8")
                checksum = hashlib.sha256(sql.encode()).hexdigest()
                if version in applied:
                    if applied[version] != checksum:
                        raise MigrationChecksumError(f"migration {version} checksum changed")
                    continue
                migration_sql = (
                    "BEGIN IMMEDIATE;\n"
                    f"{sql}\n"
                    "INSERT INTO schema_migrations VALUES "
                    f"({version}, '{checksum}', '{datetime.now(UTC).isoformat()}');\n"
                    "COMMIT;"
                )
                try:
                    connection.executescript(migration_sql)
                except sqlite3.Error:
                    connection.rollback()
                    raise
