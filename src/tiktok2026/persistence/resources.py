from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tiktok2026.contracts import AuditEvent, ResourceReservation, ResourceState, ResourceUsage
from tiktok2026.persistence.migrations import MigrationRunner, application_migrations_path
from tiktok2026.persistence.repositories import PersistenceConflictError
from tiktok2026.policies.resources import can_reserve_iteration


class ResourceLedger:
    def __init__(self, database: Path, initial: ResourceState) -> None:
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        MigrationRunner(self.database, application_migrations_path()).apply()
        with sqlite3.connect(self.database) as connection:
            self._adopt_legacy_schema(connection)
            connection.execute(
                "INSERT INTO authority_resource_state (id, payload_json) VALUES (1, ?) "
                "ON CONFLICT(id) DO NOTHING",
                (initial.model_dump_json(),),
            )

    @staticmethod
    def _adopt_legacy_schema(connection: sqlite3.Connection) -> None:
        state = connection.execute(
            "SELECT 1 FROM authority_resource_state WHERE id = 1"
        ).fetchone()
        if state is None:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(resource_state)").fetchall()
            }
            source_column = "payload_json" if "payload_json" in columns else "payload"
            if source_column in columns:
                legacy_state = connection.execute(
                    f"SELECT {source_column} FROM resource_state WHERE id = 1"
                ).fetchone()
                if legacy_state is not None:
                    ResourceState.model_validate_json(legacy_state[0])
                    connection.execute(
                        "INSERT INTO authority_resource_state (id, payload_json) VALUES (1, ?)",
                        (legacy_state[0],),
                    )
        for old_table in ("reservations", "resource_reservations"):
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (old_table,)
            ).fetchone():
                continue
            rows = connection.execute(f"SELECT * FROM {old_table}").fetchall()
            for row in rows:
                if old_table == "resource_reservations":
                    connection.execute(
                        "INSERT OR IGNORE INTO authority_resource_reservations "
                        "(reservation_id, reservation_json, status, settled_usage_json, "
                        "created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (row[0], row[1], row[2], row[3], row[4]),
                    )
                else:
                    connection.execute(
                        "INSERT OR IGNORE INTO authority_resource_reservations "
                        "(reservation_id, reservation_json, status, settled_usage_json, "
                        "created_at) "
                        "VALUES (?, ?, 'reserved', NULL, CURRENT_TIMESTAMP)",
                        (row[0], row[1]),
                    )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _read_state(connection: sqlite3.Connection) -> ResourceState:
        row = connection.execute(
            "SELECT payload_json FROM authority_resource_state WHERE id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("resource state is not initialized")
        return ResourceState.model_validate_json(row[0])

    @staticmethod
    def _write_state(connection: sqlite3.Connection, state: ResourceState) -> None:
        connection.execute(
            "UPDATE authority_resource_state SET payload_json = ? WHERE id = 1",
            (state.model_dump_json(),),
        )

    @staticmethod
    def _record_operation(
        connection: sqlite3.Connection,
        reservation: ResourceReservation,
        operation: str,
        usage: ResourceUsage | None,
        created_at: str,
        operation_id: str | None = None,
    ) -> None:
        operation_id = operation_id or f"resource-{operation}-{reservation.reservation_id}"
        usage_json = usage.model_dump_json() if usage is not None else None
        existing = connection.execute(
            "SELECT usage_json FROM authority_resource_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != usage_json:
                raise PersistenceConflictError(f"resource operation {operation_id} content changed")
            return
        connection.execute(
            "INSERT INTO authority_resource_operations "
            "(operation_id, reservation_id, operation, usage_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (operation_id, reservation.reservation_id, operation, usage_json, created_at),
        )
        event = AuditEvent(
            event_id=operation_id,
            run_id=reservation.run_id,
            experiment_id=reservation.experiment_id,
            event_type=f"resource_{operation}",
            actor_type="controller",
            actor_id="resource-ledger",
            payload={
                "reservation_id": reservation.reservation_id,
                "usage": usage.model_dump(mode="json") if usage is not None else None,
            },
            created_at=datetime.fromisoformat(created_at),
        )
        payload = event.model_dump_json()
        audit_row = connection.execute(
            "SELECT payload_json FROM audit_events WHERE event_id = ?", (operation_id,)
        ).fetchone()
        if audit_row is not None and audit_row[0] != payload:
            raise PersistenceConflictError(f"audit event {operation_id} content changed")
        if audit_row is None:
            connection.execute(
                "INSERT INTO audit_events "
                "(event_id, run_id, experiment_id, event_type, actor_type, actor_id, payload_json, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.run_id,
                    event.experiment_id,
                    event.event_type,
                    event.actor_type,
                    event.actor_id,
                    payload,
                    event.created_at.isoformat(),
                ),
            )

    def state(self) -> ResourceState:
        with sqlite3.connect(self.database) as connection:
            return self._read_state(connection)

    def reserve(self, reservation: ResourceReservation) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT reservation_json, status FROM authority_resource_reservations "
                "WHERE reservation_id = ?",
                (reservation.reservation_id,),
            ).fetchone()
            payload = reservation.model_dump_json()
            if existing is not None:
                if existing[0] != payload:
                    raise PersistenceConflictError(
                        f"reservation {reservation.reservation_id} content changed"
                    )
                connection.rollback()
                return True
            active_run = connection.execute(
                "SELECT run_id FROM authority_resource_ledger_runs WHERE id = 1"
            ).fetchone()
            if active_run is not None and active_run[0] != reservation.run_id:
                connection.rollback()
                return False
            state = self._read_state(connection)
            if reservation.purpose == "final":
                allowed = (
                    reservation.gpu_hours <= state.reserved_final_gpu_hours
                    and reservation.wall_seconds <= state.remaining_wall_seconds
                    and reservation.tokens <= state.remaining_tokens
                    and reservation.disk_bytes <= state.disk_bytes_available
                )
            else:
                allowed = can_reserve_iteration(
                    state,
                    reservation.gpu_hours,
                    reservation.wall_seconds,
                    reservation.tokens,
                    reservation.disk_bytes,
                ).allowed
            if not allowed:
                connection.rollback()
                return False
            updated = state.model_copy(
                update={
                    "remaining_gpu_hours": state.remaining_gpu_hours - reservation.gpu_hours,
                    "remaining_wall_seconds": state.remaining_wall_seconds
                    - reservation.wall_seconds,
                    "remaining_tokens": state.remaining_tokens - reservation.tokens,
                    "disk_bytes_available": state.disk_bytes_available - reservation.disk_bytes,
                    "reserved_final_gpu_hours": state.reserved_final_gpu_hours
                    - (reservation.gpu_hours if reservation.purpose == "final" else 0.0),
                }
            )
            now = datetime.now(UTC).isoformat()
            if existing is None:
                connection.execute(
                "INSERT INTO authority_resource_reservations "
                    "(reservation_id, reservation_json, status, created_at) "
                    "VALUES (?, ?, 'reserved', ?)",
                    (reservation.reservation_id, payload, now),
                )
            if active_run is None:
                connection.execute(
                    "INSERT INTO authority_resource_ledger_runs (id, run_id) VALUES (1, ?)",
                    (reservation.run_id,),
                )
            else:
                connection.execute(
                    "UPDATE authority_resource_reservations SET status = 'reserved' "
                    "WHERE reservation_id = ?",
                    (reservation.reservation_id,),
                )
            self._write_state(connection, updated)
            attempt = connection.execute(
                "SELECT COUNT(*) FROM authority_resource_operations WHERE reservation_id = ? "
                "AND operation = 'reserve'",
                (reservation.reservation_id,),
            ).fetchone()[0]
            self._record_operation(
                connection,
                reservation,
                "reserve",
                None,
                now,
                f"resource-reserve-{reservation.reservation_id}-{attempt + 1}",
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _usage(
        reservation: ResourceReservation,
        usage: ResourceUsage | None,
        gpu_hours: float | None,
        wall_seconds: float | None,
        tokens: int | None,
        disk_bytes: int | None,
    ) -> ResourceUsage:
        if usage is not None and any(
            value is not None for value in (gpu_hours, wall_seconds, tokens, disk_bytes)
        ):
            raise ValueError("provide either usage or resource keyword values")
        return usage or ResourceUsage(
            gpu_hours=reservation.gpu_hours if gpu_hours is None else gpu_hours,
            wall_seconds=reservation.wall_seconds if wall_seconds is None else wall_seconds,
            tokens=reservation.tokens if tokens is None else tokens,
            disk_bytes=reservation.disk_bytes if disk_bytes is None else disk_bytes,
        )

    def _settle(
        self,
        reservation_id: str,
        operation: str,
        usage: ResourceUsage | None,
        gpu_hours: float | None,
        wall_seconds: float | None,
        tokens: int | None,
        disk_bytes: int | None,
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT reservation_json, status, settled_usage_json "
                "FROM authority_resource_reservations "
                "WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            reservation = ResourceReservation.model_validate_json(row[0])
            if row[1] == "consumed":
                actual = self._usage(
                    ResourceReservation.model_validate_json(row[0]),
                    usage,
                    gpu_hours,
                    wall_seconds,
                    tokens,
                    disk_bytes,
                )
                if row[2] == actual.model_dump_json():
                    connection.rollback()
                    return True
                raise PersistenceConflictError(
                    f"resource settlement {reservation_id} content changed"
                )
            if row[1] != "reserved":
                connection.rollback()
                return False
            actual = self._usage(
                reservation, usage, gpu_hours, wall_seconds, tokens, disk_bytes
            )
            if (
                actual.gpu_hours > reservation.gpu_hours
                or actual.wall_seconds > reservation.wall_seconds
                or actual.tokens > reservation.tokens
                or actual.disk_bytes > reservation.disk_bytes
            ):
                raise ValueError("resource usage exceeds reservation")
            state = self._read_state(connection)
            updated = state.model_copy(
                update={
                    "remaining_gpu_hours": state.remaining_gpu_hours
                    + reservation.gpu_hours
                    - actual.gpu_hours,
                    "remaining_wall_seconds": state.remaining_wall_seconds
                    + reservation.wall_seconds
                    - actual.wall_seconds,
                    "remaining_tokens": state.remaining_tokens + reservation.tokens - actual.tokens,
                    "disk_bytes_available": state.disk_bytes_available
                    + reservation.disk_bytes
                    - actual.disk_bytes,
                    "accumulated_gpu_hours": state.accumulated_gpu_hours + actual.gpu_hours,
                    "accumulated_wall_seconds": state.accumulated_wall_seconds
                    + actual.wall_seconds,
                    "used_tokens": state.used_tokens + actual.tokens,
                    "used_disk_bytes": state.used_disk_bytes + actual.disk_bytes,
                    "reserved_final_gpu_hours": state.reserved_final_gpu_hours
                    + (
                        reservation.gpu_hours - actual.gpu_hours
                        if reservation.purpose == "final"
                        else 0.0
                    ),
                }
            )
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE authority_resource_reservations "
                "SET status = 'consumed', settled_usage_json = ? "
                "WHERE reservation_id = ?",
                (actual.model_dump_json(), reservation_id),
            )
            self._write_state(connection, updated)
            self._record_operation(connection, reservation, operation, actual, now)
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def consume(
        self,
        reservation_id: str,
        usage: ResourceUsage | None = None,
        *,
        gpu_hours: float | None = None,
        wall_seconds: float | None = None,
        tokens: int | None = None,
        disk_bytes: int | None = None,
    ) -> bool:
        return self._settle(
            reservation_id, "consume", usage, gpu_hours, wall_seconds, tokens, disk_bytes
        )

    def reconcile(
        self,
        reservation_id: str,
        usage: ResourceUsage | None = None,
        *,
        gpu_hours: float | None = None,
        wall_seconds: float | None = None,
        tokens: int | None = None,
        disk_bytes: int | None = None,
    ) -> bool:
        return self._settle(
            reservation_id, "reconcile", usage, gpu_hours, wall_seconds, tokens, disk_bytes
        )

    def release(self, reservation_id: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT reservation_json, status FROM authority_resource_reservations "
                "WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            if row[1] == "released":
                connection.rollback()
                return True
            if row[1] != "reserved":
                connection.rollback()
                return False
            reservation = ResourceReservation.model_validate_json(row[0])
            state = self._read_state(connection)
            updated = state.model_copy(
                update={
                    "remaining_gpu_hours": state.remaining_gpu_hours + reservation.gpu_hours,
                    "remaining_wall_seconds": state.remaining_wall_seconds
                    + reservation.wall_seconds,
                    "remaining_tokens": state.remaining_tokens + reservation.tokens,
                    "disk_bytes_available": state.disk_bytes_available + reservation.disk_bytes,
                    "reserved_final_gpu_hours": state.reserved_final_gpu_hours
                    + (
                        reservation.gpu_hours
                        if reservation.purpose == "final"
                        else 0.0
                    ),
                }
            )
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE authority_resource_reservations "
                "SET status = 'released' WHERE reservation_id = ?",
                (reservation_id,),
            )
            self._write_state(connection, updated)
            self._record_operation(connection, reservation, "release", None, now)
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
