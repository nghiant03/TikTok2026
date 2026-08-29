from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tiktok2026.contracts import AuditEvent, ExperimentSpec


class ApplicationRepository:
    def __init__(self, database: Path) -> None:
        self.database = database

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    parent_experiment_id TEXT,
                    status TEXT NOT NULL,
                    source_commit TEXT,
                    spec_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (parent_experiment_id) REFERENCES experiments(experiment_id)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    experiment_id TEXT,
                    event_type TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS final_test_access (
                    run_id TEXT PRIMARY KEY,
                    claimed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (kind, record_id)
                );
                """
            )

    def put_experiment(self, spec: ExperimentSpec, status: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments "
                "(experiment_id, hypothesis_id, parent_experiment_id, status, source_commit, "
                "spec_json, created_at, updated_at) VALUES (?, ?, ?, ?, NULL, ?, ?, ?) "
                "ON CONFLICT(experiment_id) DO UPDATE SET "
                "status=excluded.status, spec_json=excluded.spec_json, "
                "updated_at=excluded.updated_at",
                (
                    spec.experiment_id,
                    spec.hypothesis_id,
                    spec.parent_experiment_id,
                    status,
                    spec.model_dump_json(),
                    now,
                    now,
                ),
            )

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT spec_json FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return ExperimentSpec.model_validate_json(row[0]) if row else None

    def put_audit_event(self, event: AuditEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO audit_events "
                "(event_id, run_id, experiment_id, event_type, actor_type, actor_id, payload_json, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(event_id) DO NOTHING",
                (
                    event.event_id,
                    event.run_id,
                    event.experiment_id,
                    event.event_type,
                    event.actor_type,
                    event.actor_id,
                    event.model_dump_json(),
                    event.created_at.isoformat(),
                ),
            )

    def list_audit_events(self, run_id: str) -> tuple[AuditEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM audit_events WHERE run_id = ? "
                "ORDER BY created_at, event_id",
                (run_id,),
            ).fetchall()
        return tuple(AuditEvent.model_validate_json(row[0]) for row in rows)

    def claim_final_test_access(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO final_test_access (run_id) VALUES (?) ON CONFLICT(run_id) DO NOTHING",
                (run_id,),
            )
            return cursor.rowcount == 1

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?) "
                "ON CONFLICT(kind, record_id) DO UPDATE SET payload_json=excluded.payload_json",
                (kind, record_id, payload_json),
            )

    def list_json(self, kind: str) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM records WHERE kind = ? ORDER BY record_id", (kind,)
            ).fetchall()
        return tuple(row[0] for row in rows)
