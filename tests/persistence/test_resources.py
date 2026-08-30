import json
import sqlite3
from pathlib import Path

import pytest

from tiktok2026.contracts import ResourceReservation, ResourceState, ResourceUsage
from tiktok2026.persistence.resources import ResourceLedger


def state() -> ResourceState:
    return ResourceState(
        remaining_gpu_hours=1.0,
        accumulated_gpu_hours=0.0,
        remaining_wall_seconds=100.0,
        used_tokens=0,
        remaining_tokens=100,
        disk_bytes_available=1000,
        reserved_final_gpu_hours=0.25,
    )


def test_reservation_consumes_and_releases_resources(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / "resources.sqlite3", state())
    reservation = ResourceReservation(
        reservation_id="reservation-1",
        run_id="run-1",
        experiment_id="exp-1",
        gpu_hours=0.5,
        wall_seconds=10.0,
        tokens=10,
        disk_bytes=100,
    )
    assert ledger.reserve(reservation)
    assert ledger.reserve(reservation)
    ledger.release("reservation-1")
    assert ledger.reserve(reservation)
    assert ledger.state().remaining_gpu_hours == 1.0


def test_final_reservation_is_taken_from_protected_budget(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / "resources.sqlite3", state())
    reservation = ResourceReservation(
        reservation_id="final-reservation",
        run_id="run-1",
        gpu_hours=0.25,
        wall_seconds=5.0,
        tokens=2,
        disk_bytes=20,
        purpose="final",
    )
    assert ledger.reserve(reservation)
    assert ledger.state().reserved_final_gpu_hours == 0.0
    assert ledger.consume(
        reservation.reservation_id,
        gpu_hours=0.2,
        wall_seconds=3.0,
        tokens=1,
        disk_bytes=10,
    )
    assert ledger.state().reserved_final_gpu_hours == pytest.approx(0.05)


def test_consume_and_reconcile_account_actual_usage(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / "resources.sqlite3", state())
    reservation = ResourceReservation(
        reservation_id="reservation-1",
        run_id="run-1",
        experiment_id="exp-1",
        gpu_hours=0.5,
        wall_seconds=10.0,
        tokens=10,
        disk_bytes=100,
    )
    assert ledger.reserve(reservation)
    assert ledger.consume(
        reservation.reservation_id,
        ResourceUsage(gpu_hours=0.25, wall_seconds=4.0, tokens=3, disk_bytes=40),
    )
    current = ledger.state()
    assert current.remaining_gpu_hours == 0.75
    assert current.accumulated_gpu_hours == 0.25
    assert current.accumulated_wall_seconds == 4.0
    assert current.used_tokens == 3
    assert current.used_disk_bytes == 40
    assert ledger.reconcile(
        reservation.reservation_id,
        ResourceUsage(gpu_hours=0.25, wall_seconds=4.0, tokens=3, disk_bytes=40),
    )


def test_overage_settlement_consumes_reservation_without_stranding_it(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / "resources.sqlite3", state())
    reservation = ResourceReservation(
        reservation_id="reservation-overage",
        run_id="run-1",
        experiment_id="exp-1",
        gpu_hours=0.1,
        wall_seconds=1.0,
        tokens=1,
        disk_bytes=10,
    )
    assert ledger.reserve(reservation)
    actual = ResourceUsage(gpu_hours=0.2, wall_seconds=2.0, tokens=2, disk_bytes=20)
    assert ledger.consume(reservation.reservation_id, actual)
    assert ledger.reconcile(reservation.reservation_id, actual)
    with sqlite3.connect(ledger.database) as connection:
        assert connection.execute(
            "SELECT status FROM authority_resource_reservations WHERE reservation_id = ?",
            (reservation.reservation_id,),
        ).fetchone()[0] == "consumed"


def test_new_run_reclaims_interrupted_reservation(tmp_path: Path) -> None:
    ledger = ResourceLedger(tmp_path / "resources.sqlite3", state())
    ledger.claim_run("run-1")
    assert ledger.reserve(
        ResourceReservation(
            reservation_id="stale-reservation",
            run_id="run-1",
            gpu_hours=0.5,
            wall_seconds=10.0,
            tokens=10,
            disk_bytes=100,
        )
    )

    ledger.claim_run("run-2")

    assert ledger.state() == state()
    assert ledger.reserve(
        ResourceReservation(
            reservation_id="new-reservation",
            run_id="run-2",
            gpu_hours=0.5,
            wall_seconds=10.0,
            tokens=10,
            disk_bytes=100,
        )
    )
    assert ledger.release_run("run-2")
    assert ledger.state() == state()


def test_legacy_resource_ledger_schema_is_migrated(tmp_path: Path) -> None:
    database = tmp_path / "legacy-resources.sqlite3"
    legacy_payload = json.dumps(
        {
            "schema_version": "1",
            "remaining_gpu_hours": 0.5,
            "accumulated_gpu_hours": 0.5,
            "remaining_wall_seconds": 90.0,
            "used_tokens": 10,
            "remaining_tokens": 90,
            "disk_bytes_available": 900,
            "reserved_final_gpu_hours": 0.25,
        }
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE resource_state "
            "(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE reservations (reservation_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        connection.execute("INSERT INTO resource_state VALUES (1, ?)", (legacy_payload,))
    ledger = ResourceLedger(database, state())

    assert ledger.state().remaining_gpu_hours == 0.5
    assert ledger.reserve(
        ResourceReservation(
            reservation_id="legacy-follow-up",
            run_id="run-1",
            gpu_hours=0.1,
            wall_seconds=1.0,
            tokens=1,
            disk_bytes=10,
        )
    )
