from pathlib import Path

from tiktok2026.contracts import ResourceReservation, ResourceState
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
    assert not ledger.reserve(reservation)
    ledger.release("reservation-1")
    assert ledger.reserve(reservation)
