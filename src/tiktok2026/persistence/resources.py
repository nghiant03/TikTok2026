from __future__ import annotations

import sqlite3
from pathlib import Path

from tiktok2026.contracts import ResourceReservation, ResourceState
from tiktok2026.policies.resources import can_reserve_iteration


class ResourceLedger:
    def __init__(self, database: Path, initial: ResourceState) -> None:
        self.database = database
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS resource_state "
                "(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reservations "
                "(reservation_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO resource_state VALUES (1, ?) ON CONFLICT(id) DO NOTHING",
                (initial.model_dump_json(),),
            )

    def state(self) -> ResourceState:
        with sqlite3.connect(self.database) as connection:
            row = connection.execute("SELECT payload FROM resource_state WHERE id=1").fetchone()
        if row is None:
            raise RuntimeError("resource state is not initialized")
        return ResourceState.model_validate_json(row[0])

    def reserve(self, reservation: ResourceReservation) -> bool:
        connection = sqlite3.connect(self.database, isolation_level=None)
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM reservations WHERE reservation_id=?", (reservation.reservation_id,)
            ).fetchone()
            if existing:
                connection.rollback()
                return False
            state_row = connection.execute(
                "SELECT payload FROM resource_state WHERE id=1"
            ).fetchone()
            if state_row is None:
                connection.rollback()
                return False
            state = ResourceState.model_validate_json(state_row[0])
            decision = can_reserve_iteration(
                state,
                reservation.gpu_hours,
                reservation.wall_seconds,
                reservation.tokens,
                reservation.disk_bytes,
            )
            if not decision.allowed:
                connection.rollback()
                return False
            updated = state.model_copy(
                update={
                    "remaining_gpu_hours": state.remaining_gpu_hours - reservation.gpu_hours,
                    "remaining_wall_seconds": state.remaining_wall_seconds
                    - reservation.wall_seconds,
                    "remaining_tokens": state.remaining_tokens - reservation.tokens,
                    "disk_bytes_available": state.disk_bytes_available - reservation.disk_bytes,
                }
            )
            connection.execute(
                "INSERT INTO reservations VALUES (?, ?)",
                (reservation.reservation_id, reservation.model_dump_json()),
            )
            connection.execute(
                "UPDATE resource_state SET payload=? WHERE id=1", (updated.model_dump_json(),)
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def release(self, reservation_id: str) -> None:
        connection = sqlite3.connect(self.database, isolation_level=None)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return
            reservation = ResourceReservation.model_validate_json(row[0])
            state_row = connection.execute(
                "SELECT payload FROM resource_state WHERE id=1"
            ).fetchone()
            if state_row is None:
                connection.rollback()
                return
            state = ResourceState.model_validate_json(state_row[0])
            updated = state.model_copy(
                update={
                    "remaining_gpu_hours": state.remaining_gpu_hours + reservation.gpu_hours,
                    "remaining_wall_seconds": state.remaining_wall_seconds
                    + reservation.wall_seconds,
                    "remaining_tokens": state.remaining_tokens + reservation.tokens,
                    "disk_bytes_available": state.disk_bytes_available + reservation.disk_bytes,
                }
            )
            connection.execute("DELETE FROM reservations WHERE reservation_id=?", (reservation_id,))
            connection.execute(
                "UPDATE resource_state SET payload=? WHERE id=1", (updated.model_dump_json(),)
            )
            connection.commit()
        finally:
            connection.close()
